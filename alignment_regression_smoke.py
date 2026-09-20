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


if __name__ == '__main__':
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
    try:
        result = run_case(app)
        Path(os.environ['METEOR_ALIGNMENT_SMOKE_REPORT']).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        app.destroy()
