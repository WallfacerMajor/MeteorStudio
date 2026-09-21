"""Real Tk + Siril + PTGui regression driven by a caller-provided case manifest."""
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile


def run_case(app):
    if os.environ.get("METEOR_ALIGNMENT_FOCAL_SMOKE"):
        return run_focal_case(app)
    from toolbox_smoke import click, pump
    from ptgui_pipeline import siril_find_stars
    case = json.loads(Path(os.environ['METEOR_ALIGNMENT_CASE']).read_text(encoding='utf-8'))
    output = Path(os.environ['METEOR_ALIGNMENT_OUTPUT']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(item['source']) for item in case['items']]
    reference = Path(case['base_layer'])
    export_layers = os.environ.get('METEOR_ALIGNMENT_EXPORT') == '1'

    def digest(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()

    hashes = {path: digest(path) for path in [reference, *sources]}
    # Independently verify the real executable's coordinate convention against
    # known image-space star centres, not a mock of the old parser assumption.
    yy, xx = np.mgrid[:320, :440]
    centres = np.float32([(55, 43), (145, 72), (270, 105), (320, 218), (105, 240)])
    image = 2000. + np.random.default_rng(41).normal(0, 35, xx.shape)
    for x, y in centres:
        image += 28000*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.8**2))
    proxy = output/'known-star-centres.png'
    cv2.imencode('.png', image.astype(np.uint16))[1].tofile(proxy)
    detected, _ = siril_find_stars(Path(case['siril_path']), proxy, output/'known-stars', 'known')
    distances = np.linalg.norm(centres[:, None]-detected[None, :], axis=2).min(axis=1)
    assert float(distances.max()) < 1.0, (centres, detected)

    app.show_toolbox()
    pump(app, .3)
    click(app, app.toolbox_home.tool_buttons['alignment' if export_layers else 'control_points'])
    window = app.alignment_window
    window.base_path.set(str(reference))
    window.meteor_dir.set(str(sources[0].parent))
    window.output_dir.set(str(output))
    errors = []

    def wait_until(predicate, timeout=600):
        deadline = time.monotonic()+timeout
        while not predicate():
            pump(app, .05)
            assert not errors, errors
            assert time.monotonic() < deadline, window.status.get()

    with patch('alignment_workspace.show_copyable_error', side_effect=lambda *args, **kw: errors.append(str(args))):
        click(app, window.scan_button)
        wait_until(lambda: not window.running)
        assert set(window.items) == set(sources)
        click(app, window.run_button)
        wait_until(lambda: not window.running)
        assert not errors and window.last_result
    result = window.last_result
    for item in result.items:
        assert item.control_points >= 6 and item.median_error <= 8, asdict(item)
        assert item.status.startswith('已导出' if export_layers else '工程已生成'), asdict(item)
        if export_layers:
            assert Path(item.output_layer) != Path(item.source)
            with tifffile.TiffFile(item.output_layer) as image:
                assert image.pages[0].dtype == np.dtype('uint16')
    projects = list((Path(result.project_dir)/'ptgui_project/ready_projects').glob('*.pts'))
    assert len(projects) == len(sources)
    for project in projects:
        data = json.loads(project.read_text(encoding='utf-8'))['project']
        assert len(data['imagegroups']) == 2
        assert len(data['controlpoints']) >= 6
    assert all(digest(path) == value for path, value in hashes.items())
    payload = asdict(result)
    payload['checks'] = dict.fromkeys(('real_siril_top_left_coordinates', 'real_tk_scan_and_run',
        'all_current_images_aligned', 'independent_ptgui_projects', 'input_hashes_unchanged'), 'passed')
    if export_layers:
        payload['checks']['real_16bit_layers'] = 'passed'
    window.destroy()
    return payload


def run_focal_case(app):
    import tempfile
    from PIL import Image
    from tkinter import ttk
    from toolbox_smoke import click, pump, widgets
    from ptgui_pipeline import AlignmentResult
    errors, calls = [], []
    checks = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        inputs = root/'inputs'
        inputs.mkdir()
        plain = root/'no-exif.png'
        tagged = root/'exif.jpg'
        Image.new('RGB', (120, 80)).save(plain)
        Image.new('RGB', (120, 80)).save(inputs/'frame.png')
        exif = Image.Exif()
        exif[37386] = 24.0
        Image.new('RGB', (120, 80)).save(tagged, exif=exif)
        for mode in ('alignment', 'control_points'):
            app.show_toolbox()
            click(app, app.toolbox_home.tool_buttons[mode])
            window = app.alignment_window
            window.geometry('960x700')
            pump(app, .3)
            window.meteor_dir.set(str(inputs))
            window.output_dir.set(str(root/'output'))
            picker = next(w for w in widgets(window.input_panel) if isinstance(w, ttk.Button))
            with patch('alignment_workspace.filedialog.askopenfilename', return_value=str(plain)):
                click(app, picker)
            field = window.reference_focal_input
            def visible():
                pump(app, .15)
                assert window.configuration.select() == str(window.inputs_tab)
                x = field.winfo_rootx()+field.winfo_width()//2
                y = field.winfo_rooty()+field.winfo_height()//2
                assert window.winfo_containing(x, y) == field, 'Focal entry is obscured'
                canvas = window.inputs_tab._inspector_canvas
                assert field.winfo_rooty()+field.winfo_height() <= canvas.winfo_rooty()+canvas.winfo_height()
                assert field.instate(['!disabled'])
            visible()
            with patch('alignment_workspace.show_copyable_error', side_effect=lambda *a, **k: errors.append(a)):
                click(app, window.scan_button)
                visible()
                assert not window.running and not errors
                # Real mouse and key events, not StringVar.set, enter the value.
                click(app, field)
                field.focus_force()
                field.event_generate('<KeyPress>', keysym='0')
                click(app, window.scan_button)
                visible()
                assert not window.running and not errors
                field.focus_force()
                field.event_generate('<KeyPress>', keysym='BackSpace')
                for key in '35':
                    field.event_generate('<KeyPress>', keysym=key)
                    field.event_generate('<KeyRelease>', keysym=key)
                assert field.get() == '35', field.get()
                click(app, window.scan_button)
                deadline = time.monotonic()+5
                while window.running and time.monotonic()<deadline:
                    pump(app, .05)
                assert len(window.items)==1 and window.run_button.instate(['!disabled'])
                def pipeline(*args, **kwargs):
                    calls.append(kwargs)
                    return AlignmentResult(str(root), "", str(plain), None, [], "", "", "", "")
                with patch('toolbox.require_software', return_value={'ptgui':Path('ptgui'), 'siril':Path('siril')}), patch('alignment_workspace.run_alignment_pipeline', side_effect=pipeline), patch('alignment_workspace.messagebox.askyesno', return_value=False):
                    click(app, window.run_button)
                    pump(app, .5)
                assert calls[-1]['reference_focal_length']==35 and not errors
            with patch('alignment_workspace.filedialog.askopenfilename', return_value=str(tagged)):
                click(app, picker)
            assert field.instate(['disabled']) and float(field.get())==24
            with patch('alignment_workspace.filedialog.askopenfilename', return_value=str(plain)):
                click(app, picker)
            visible()
            assert field.get()==''
            # Switch actual tabs with pointer events and inspect the rendered selection.
            from PIL import ImageGrab
            notebook = window.configuration
            for index in (1, 0):
                x = next(x for x in range(3, notebook.winfo_width(), 3)
                         if notebook.identify(x, 10) and notebook.index(f'@{x},10') == index)
                notebook.event_generate('<ButtonPress-1>', x=x+5, y=10)
                notebook.event_generate('<ButtonRelease-1>', x=x+5, y=10)
                pump(app, .2)
                assert notebook.index(notebook.select()) == index
                left, top = notebook.winfo_rootx(), notebook.winfo_rooty()
                picture = ImageGrab.grab((left, top, left+notebook.winfo_width(), top+notebook.winfo_height()))
                picture.save(Path(os.environ['METEOR_ALIGNMENT_SMOKE_REPORT']).with_name('tabs-debug.png'))
                positions = [px for py in range(min(12, picture.height)) for px in range(picture.width)
                             if picture.getpixel((px, py))[:3] == (129,174,232)]
                assert positions and notebook.index(f'@{int(sum(positions)/len(positions))},10') == index
                if os.environ.get('METEOR_ALIGNMENT_SMOKE_REPORT'):
                    picture.save(Path(os.environ['METEOR_ALIGNMENT_SMOKE_REPORT']).with_name(f'tabs-{mode}-{index}.png'))
            checks.append(mode+': visible input, keyboard entry, scan/run parameter, EXIF switching, tab selection pixels')
            window.destroy()
            app.alignment_window = None
    return {'checks': checks, 'errors': errors}


if __name__ == '__main__':
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
    try:
        result = run_case(app)
        Path(os.environ['METEOR_ALIGNMENT_SMOKE_REPORT']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        app.destroy()
