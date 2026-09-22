"""Real Tk regression: a mask adjustment must not replace recommended blend values."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from tkinter import ttk

from meteor_composer import MeteorComposer, Stroke, compose_meteor_objects


def run():
    with tempfile.TemporaryDirectory() as directory, patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
        app.autosave_path = Path(directory) / 'autosave.json'
        app._schedule_autosave = lambda: None
        try:
            app.geometry('1350x920+0+0')
            app.show_composite_workspace()
            base = np.full((300, 480, 3), 25, np.uint8)
            source = base.copy()
            cv2.line(source, (80, 170), (400, 110), (180, 210, 240), 5)
            cv2.circle(source, (220, 158), 2, (255, 255, 255), -1)
            key = str(Path(directory) / 'source.tif')
            stroke = Stroke([(80/479, 170/299), (400/479, 110/299)], 28, 7,
                            star_removal=40, auto_black_point=2.0,
                            auto_cleanup=80, auto_brightness=105, auto_feather=7)
            app.files = [Path(key)]
            app.pairs = {key: Path(directory) / 'base.tif'}
            app.current_path = Path(key)
            app.current_dims = (480, 300)
            app.preview_base = base
            app.preview_source = source
            app.preview_aligned_source = source
            app.preview_original_source = source
            app.strokes = {key: [stroke]}
            app.adjustment_defaults.update(auto_optimize=True, match_exposure=False,
                                           curve_enabled=False)
            app.blend_mode.set('线性减淡（添加）')
            app.output_mode.set('combined')
            app.global_preview_rgb, _ = compose_meteor_objects(
                source, base, [stroke], *app._object_composite_settings(key)
            )
            app.global_preview_signature = app._global_preview_state_signature()
            app.exact_preview_full_rgb = app.global_preview_rgb
            app.exact_preview_signature = app._exact_preview_state_signature()
            app.selected_object = (key, 0)
            app.view_mode.set('blend')
            app._render_preview()
            app.control_notebook.select(app.selected_tools_tab)
            app._load_selected_object_adjustments()
            app.update()
            assert app.selected_override_enabled.get()
            assert app.selected_brightness.get() == 105
            assert app.selected_cleanup.get() == 80
            star_scale = next(widget for widget in app.selected_object_controls
                              if isinstance(widget, ttk.Scale)
                              and str(widget.cget('variable')) == str(app.selected_star_removal))
            x, y = star_scale.coords()
            star_scale.event_generate('<ButtonPress-1>', x=int(x), y=int(y))
            star_scale.event_generate('<B1-Motion>', x=int(star_scale.winfo_width()*.65),
                                      y=int(y), state=0x100)
            star_scale.event_generate('<ButtonRelease-1>',
                                      x=int(star_scale.winfo_width()*.65), y=int(y))
            deadline = time.monotonic() + 8
            while getattr(app, '_parameter_busy', False) or getattr(app, '_parameter_pending', {}):
                app.update()
                time.sleep(.01)
                assert time.monotonic() < deadline
            app.update()
            assert stroke.star_removal != 40
            assert stroke.brightness_override is None
            assert stroke.background_cleanup_override is None
            expected_mask_edit, _ = compose_meteor_objects(
                source, base, [stroke], *app._object_composite_settings(key)
            )
            try:
                np.testing.assert_array_equal(app.global_preview_rgb, expected_mask_edit)
            except AssertionError:
                ys, xs = np.nonzero(np.any(app.global_preview_rgb != expected_mask_edit, axis=2))
                print('mask mismatch', app.last_incremental_box,
                      (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                      len(xs), flush=True)
                raise

            # Re-run the recommendation after manual tuning. Its result must
            # become effective instead of remaining hidden by old overrides.
            stroke.brightness_override = 180
            stroke.background_cleanup_override = 100
            app.global_preview_rgb, _ = compose_meteor_objects(
                source, base, [stroke], *app._object_composite_settings(key)
            )
            app.global_preview_signature = app._global_preview_state_signature()
            parameters = dict(strength='标准', black_point=1.5,
                              cleanup=72, brightness=108, feather=7)
            app.work_queue.put(('blend_optimized', [
                (key, 0, stroke.points.copy(), parameters),
            ], '标准', 1))
            app._poll_queue()
            assert stroke.brightness_override is None
            assert stroke.background_cleanup_override is None
            assert app.selected_brightness.get() == 108
            assert app.selected_cleanup.get() == 72
            expected, _ = compose_meteor_objects(
                source, base, [stroke], *app._object_composite_settings(key)
            )
            np.testing.assert_array_equal(app.global_preview_rgb, expected)
            deadline = time.monotonic() + 1.4
            while time.monotonic() < deadline:
                app.update()
                time.sleep(.01)
            return {'mask_slider_preserves_recommendation': True,
                    'rerun_recommendation_clears_manual_override': True,
                    'preview_matches_compositor': True}
        finally:
            app.destroy()


if __name__ == '__main__':
    print(json.dumps(run()))
