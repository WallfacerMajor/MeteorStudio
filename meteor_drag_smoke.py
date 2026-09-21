"""Real pointer-event regression for non-current, smaller original layers."""
import json
import tempfile
import time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import cv2
import numpy as np
import tifffile


def run_smoke(app):
    from toolbox_smoke import pump, widgets, click
    from meteor_composer import Stroke, compose_meteor_sources, place_source_on_canvas
    app.show_composite_workspace()
    app.geometry('1500x1000+0+0')
    pump(app, .5)
    def wait_ready():
        deadline = time.monotonic() + 30
        while app.preview_base is None or app.exact_preview_signature != app._exact_preview_state_signature():
            pump(app, .05)
            assert time.monotonic() < deadline, app.status.get()
        pump(app, 1.65)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        app.autosave_path = root / 'autosave.json'
        inputs = root / 'inputs'; inputs.mkdir()
        base = np.full((400, 600, 3), 3000, np.uint16)
        original = np.full((260, 400, 3), 3000, np.uint16)
        cv2.line(original, (90, 110), (210, 140), (42000, 36000, 30000), 3)
        tifffile.imwrite(root / 'base.tif', base)
        tifffile.imwrite(root / 'original.tif', original)
        for name in ('a', 'b'):
            tifffile.imwrite(inputs / (name + '.tif'), base)
        app.source_dir.set(str(inputs)); app.base_dir.set(str(root / 'base.tif'))
        app.output_dir.set(str(root / 'output')); app.output_mode.set('combined')
        app.scan_inputs()
        deadline = time.monotonic() + 30
        while app.current_path is None or app.preview_base is None:
            pump(app, .05)
            assert time.monotonic() < deadline, app.status.get()
        pump(app, 1.65)
        app._set_view_mode('blend')
        key = str(app.files[1])
        app.original_sources[key] = root / 'original.tif'
        app.use_original_sources.add(key)
        stroke = Stroke([(190/600, 180/400), (310/600, 210/400)], 18, 3, source_mode='original',
                        auto_blend_enabled=False)
        app.strokes = {key: [stroke]}
        app.adjustment_defaults.update(auto_optimize=False, match_exposure=False, curve_enabled=False,
                                       background_cleanup=0, meteor_brightness=100)
        checks = {}
        for mode in ('自然融合', '滤色', '线性减淡（添加）'):
            app.blend_mode.set(mode)
            app._invalidate_global_preview(); app._render_preview(); wait_ready()
            app._canvas_fit(); pump(app, .2)
            app.selected_object = (key, 0); app._draw_selected_object_overlay()
            assert app.current_path != Path(key)
            assert app._object_canvas_width(key) == 600
            for evicted in (False, True):
                if evicted:
                    with app.preview_cache_lock:
                        app.layer_preview_cache.clear()
                center = app._object_canvas_geometry((key, 0))['center']
                x, y = map(int, center)
                old = (stroke.offset_x, stroke.offset_y)
                with ExitStack() as stack:
                    monitors = {n: stack.enter_context(patch.object(app, n, wraps=getattr(app, n))) for n in
                                ('_invalidate_global_preview', '_global_preview_worker', '_exact_preview_worker')}
                    app.canvas.event_generate('<ButtonPress-1>', x=x, y=y)
                    app.canvas.event_generate('<B1-Motion>', x=x+18, y=y+9, state=0x100)
                    pump(app, .4)
                    app.canvas.event_generate('<ButtonRelease-1>', x=x+18, y=y+9)
                    moved = (stroke.offset_x, stroke.offset_y)
                    assert moved != old
                    frame = app.preview_rgb.copy()
                    pump(app, 1.7)
                    assert moved == (stroke.offset_x, stroke.offset_y)
                    np.testing.assert_array_equal(frame, app.preview_rgb)
                    assert not any(m.called for m in monitors.values()), {n:m.call_count for n,m in monitors.items()}
                    assert app.global_exact_after_id is None
                a = app.adjustment_defaults
                expected, _ = compose_meteor_sources(
                    (base >> 8).astype(np.uint8),
                    (place_source_on_canvas(original, 400, 600) >> 8).astype(np.uint8),
                    (base >> 8).astype(np.uint8), [stroke], False, False,
                    float(a['curve_shadows']), float(a['curve_highlights']), mode,
                    bool(a.get('preserve_brightness', True)), 100, 0, False,
                    background=(base >> 8).astype(np.uint8))
                np.testing.assert_allclose(app.preview_rgb, expected, atol=1)
                checks[f'{mode}_cache_{"evicted" if evicted else "present"}'] = 'passed'
        app.control_notebook.select(app.selected_tools_tab)
        pump(app, .2)
        buttons = {str(w.cget('text')):w for w in widgets(app) if getattr(w, '_action_icon', False)}
        auto = buttons['恢复自动值']; original_button = buttons['恢复原始融合']
        assert auto._action_key == 'spark' and original_button._action_key == 'reset'
        assert auto.cget('image') != original_button.cget('image')
        for button, phrase in ((auto, '恢复自动优化参数'), (original_button, '保留手动参数与位置')):
            button.event_generate('<Enter>'); pump(app, .55)
            assert phrase in button._action_hint.popup.winfo_children()[0].cget('text')
            button.event_generate('<Leave>'); pump(app, .05)
        checks['distinct_restore_icons_and_real_hover'] = 'passed'
        return checks


if __name__ == '__main__':
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
    try:
        print(json.dumps(run_smoke(app), ensure_ascii=False))
    finally:
        app.destroy()
