"""Exercise the guided path using real Tk clicks and a real TIFF export."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile


def run_smoke(app, exercise_blending=False):
    from toolbox_smoke import click, pump, widgets
    from tkinter import ttk
    from action_icons import hint_text
    app.show_composite_workspace()
    app.geometry('1500x1000+0+0')
    pump(app, .3)
    workflow = app.composite_workflow
    assert app.export_button.instate(['disabled'])
    assert app.auto_detect_button.instate(['disabled'])
    assert '扫描' in hint_text(app.export_button)
    assert all(button.master is app.file_actions for button in
               (app.load_project_button, app.save_project_button, app.export_button, app.open_output_button))
    errors = []
    def wait(predicate):
        deadline = time.monotonic()+30
        while not predicate():
            pump(app, .05)
            assert time.monotonic() < deadline, app.status.get()
            assert not errors, errors
    with tempfile.TemporaryDirectory() as folder, patch('meteor_composer.show_copyable_error', side_effect=lambda *a, **k: errors.append(a)):
        root = Path(folder)
        app.autosave_path = root/'autosave.json'
        inputs = root/'inputs'
        inputs.mkdir()
        base = np.full((400,600,3), 1800, dtype=np.uint16)
        source = base.copy()
        cv2.line(source, (150,220), (450,120), (50000,50000,50000), 5)
        tifffile.imwrite(root/'base.tif', base, photometric='rgb')
        tifffile.imwrite(inputs/'meteor.tif', source, photometric='rgb')
        app.source_dir.set(str(inputs))
        app.base_dir.set(str(root/'base.tif'))
        app.output_dir.set(str(root/'output'))
        app.output_mode.set('combined')
        app.export_tiff.set(True)
        click(app, workflow.next_button)
        wait(lambda: app.current_path is not None and app.preview_source is not None)
        pump(app, 1.5)
        assert app.export_button.instate(['disabled'])
        assert '只显示底图' in workflow.message.cget('text')
        with patch.object(app, 'auto_detect_all') as detect:
            click(app, workflow.next_button)
            detect.assert_called_once()
        # The guide also allows manual marking; it never requires AI approval.
        app._set_paths_panel_visible(False)
        pump(app, .4)
        app._canvas_fit()
        pump(app, .2)
        x0,y0,x1,y1 = app.display_box
        sx,sy = int(x0+(x1-x0)*.25),int(y0+(y1-y0)*.55)
        ex,ey = int(x0+(x1-x0)*.75),int(y0+(y1-y0)*.30)
        app.canvas.event_generate('<ButtonPress-1>', x=sx, y=sy)
        app.canvas.event_generate('<B1-Motion>', x=ex, y=ey, state=0x100)
        app.canvas.event_generate('<ButtonRelease-1>', x=ex, y=ey)
        pump(app, 1.6)
        assert app.export_button.instate(['!disabled'])
        assert workflow.stage == 1
        click(app, workflow.next_button)
        pump(app, 1.6)
        assert app.view_mode.get() == 'blend' and workflow.stage == 2
        def viewport():
            return (app.display_box, app.canvas_center_x, app.canvas_center_y,
                    app.canvas.winfo_width(), app.canvas.winfo_height())
        before = viewport()
        with patch.object(app, '_invalidate_global_preview', wraps=app._invalidate_global_preview) as invalidate, patch.object(app, '_global_preview_worker', wraps=app._global_preview_worker) as global_job, patch.object(app, '_exact_preview_worker', wraps=app._exact_preview_worker) as exact_job:
            workflow.signature = None
            workflow.refresh()
            label = workflow.steps[2]
            label.event_generate('<ButtonPress-1>', x=5, y=5)
            label.event_generate('<ButtonRelease-1>', x=5, y=5)
            pump(app, 1.6)
            assert viewport() == before
            assert not invalidate.called and not global_job.called and not exact_job.called
        import os
        if os.environ.get('METEOR_WORKFLOW_SMOKE_REPORT'):
            from PIL import ImageGrab
            x,y = app.winfo_rootx(),app.winfo_rooty()
            ImageGrab.grab((x,y,x+app.winfo_width(),y+app.winfo_height())).save(Path(os.environ['METEOR_WORKFLOW_SMOKE_REPORT']).with_suffix('.png'))
        if exercise_blending:
            from meteor_blending_smoke import check_blending
            check_blending(app, root, click, pump, wait)
        project = root/'project.json'
        with patch('meteor_composer.filedialog.asksaveasfilename', return_value=str(project)):
            click(app, app.save_project_button)
        assert project.exists() and json.loads(project.read_text(encoding='utf-8'))['strokes']
        with patch('meteor_composer.filedialog.askopenfilename', return_value=str(project)):
            click(app, app.load_project_button)
        pump(app, 1.6)
        if exercise_blending:
            restored = next(s for values in app.strokes.values() for s in values if not s.erase)
            assert restored.blend_mode_override == '线性减淡（添加）'
            assert restored.star_removal > 0 and restored.mask_choke > 0
        with patch('meteor_composer.messagebox.askyesnocancel', return_value=False), patch('meteor_composer.messagebox.askyesno', return_value=False), patch('meteor_composer.messagebox.showinfo'):
            click(app, app.export_button)
            wait(lambda: not app.export_running and app.last_export_path is not None)
        outputs = list(app.last_export_path.rglob('*.tif')) + list(app.last_export_path.rglob('*.tiff'))
        assert outputs
        assert any(tifffile.imread(path).dtype == np.uint16 and tifffile.imread(path).max()>base.max()+1000 for path in outputs)
        pump(app, .3)
        assert app.open_output_button.instate(['!disabled'])
        clear = next(w for w in widgets(app.mask_tools_tab) if isinstance(w,ttk.Button) and w.cget('text')=='清除此图蒙版')
        with patch('meteor_composer.messagebox.askyesno', return_value=True):
            click(app, clear)
        pump(app, 1.6)
        assert app.export_button.instate(['disabled'])
        assert '只显示底图' in workflow.message.cget('text')
    return dict.fromkeys(('shared_file_actions', 'missing_inputs_guard', 'empty_masks_guard',
        'guide_detection_action', 'manual_paint_unlocks_export', 'fusion_step',
        'guide_refresh_preserves_viewport_and_starts_no_pixel_workers', 'project_save_load', 'real_16bit_export_contains_meteor', 'delete_returns_to_selection'), 'passed')
