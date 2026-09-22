"""New focused live-preview check: Tk events, intermediate frames and settled pixels."""
import json
import tempfile
import time
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_preview import LivePreview
from meteor_blending import BLEND_MODES
from meteor_composer import MeteorComposer, Stroke, adjust_composite_base_exposure, compose_meteor_objects


def pump(app, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.update()
        time.sleep(.005)


def run():
    assert BLEND_MODES == ('线性减淡（添加）', '滤色')
    sample_base = np.full((128, 192, 3), 24000, np.uint16)
    sample_source = sample_base.copy()
    cv2.line(sample_source, (30, 70), (165, 54), (50000, 47000, 45000), 5)
    sample_stroke = Stroke([(30/191, 70/127), (165/191, 54/127)], 18, 4)
    images = {name: compose_meteor_objects(sample_source, sample_base, [sample_stroke],
              False, False, 0, 100, name, False, 100, 0, False)[0]
              for name in ('自然融合', '滤色', '线性减淡（添加）')}
    assert np.any(images['滤色'] != images['线性减淡（添加）'])
    single_difference = np.abs(images['自然融合'].astype(np.int32)-images['线性减淡（添加）'])
    assert single_difference.max() <= 1 and np.count_nonzero(single_difference) < 30
    overlap = {name: compose_meteor_objects(sample_source, sample_base, [sample_stroke, sample_stroke],
               False, False, 0, 100, name, False, 100, 0, False)[0]
               for name in ('自然融合', '线性减淡（添加）')}
    assert np.any(overlap['自然融合'] != overlap['线性减淡（添加）'])
    with tempfile.TemporaryDirectory() as tmp, patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
        app.autosave_path = Path(tmp) / 'autosave.json'
        app._schedule_autosave = lambda: None
        app.geometry('1350x900+0+0')
        app.show_composite_workspace()
        try:
            assert app.blend_mode.get() == '线性减淡（添加）'
            h, w = 2050, 2050
            base = np.full((h, w, 3), 85, np.uint8)
            raw = base.copy()
            cv2.line(raw, (300, 1100), (1700, 900), (200, 180, 160), 15)
            key = str(Path(tmp) / 'source.tif')
            app.current_path = Path(key)
            app.current_dims = (w, h)
            app.files = [Path(key)]
            app.pairs = {key: Path(tmp) / 'base.tif'}
            app.preview_base = base
            app.preview_source = raw
            app.preview_aligned_source = raw
            app.preview_original_source = raw
            app.strokes = {key: [Stroke([(300/w, 1100/h), (1700/w, 900/h)], 40, 5)]}
            app.candidates = {}
            app.output_mode.set('combined')
            app.view_mode.set('blend')
            app.global_preview_rgb = raw
            app.global_preview_signature = app._global_preview_state_signature()
            app.exact_preview_full_rgb = raw
            app.exact_preview_signature = app._exact_preview_state_signature()
            app._render_preview()
            pump(app, .1)
            app._canvas_fit()
            forbidden = []
            app._request_global_preview = lambda *_: forbidden.append('global')
            app._start_automatic_exact_preview = lambda *_: forbidden.append('exact')
            scales = [widget for widget in app.winfo_children()]
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            scale = next(x for x in descendants(app) if isinstance(x, ttk.Scale)
                         and str(x.cget('variable')) == str(app.base_exposure_tenths))
            parent = scale
            while not parent.winfo_ismapped():
                host = parent.nametowidget(parent.winfo_parent())
                if isinstance(host, ttk.Notebook):
                    host.select(parent)
                    pump(app, .05)
                parent = host
            assert scale.winfo_ismapped(), 'exposure control is hidden'
            app.lift()
            app.focus_force()
            scale.event_generate('<ButtonPress-1>', x=scale.winfo_width()//2, y=scale.winfo_height()//2)
            heartbeat = []
            app.after(20, lambda: heartbeat.append(time.monotonic()))
            events = []
            exposure_values = []
            for fraction in (.55, .65, .75):
                start = time.monotonic()
                scale.event_generate('<B1-Motion>', x=int(scale.winfo_width()*fraction), y=scale.winfo_height()//2, state=0x100)
                events.append(time.monotonic()-start)
                pump(app, .07)
                exposure_values.append(app.base_exposure_tenths.get())
            scale.event_generate('<ButtonRelease-1>', x=int(scale.winfo_width()*.75), y=scale.winfo_height()//2)
            pump(app, 1.5)
            assert heartbeat, 'Tk froze during slider movement'
            assert max(events) < .2, events
            assert len(set(exposure_values)) >= 2, exposure_values
            assert app.exact_preview_signature == app._exact_preview_state_signature()
            expected = adjust_composite_base_exposure(raw, base, app.base_exposure_tenths.get()/10)
            np.testing.assert_array_equal(app.exact_preview_full_rgb, expected)
            assert not forbidden, forbidden
            combo = next(x for x in descendants(app) if isinstance(x, ttk.Combobox)
                         and str(x.cget('textvariable')) == str(app.blend_mode))
            assert tuple(combo.cget('values')) == BLEND_MODES
            combo.event_generate('<ButtonPress-1>', x=combo.winfo_width()//2, y=combo.winfo_height()//2)
            combo.event_generate('<ButtonRelease-1>', x=combo.winfo_width()//2, y=combo.winfo_height()//2)
            combo.current(1)
            combo.event_generate('<<ComboboxSelected>>')
            pump(app, .1)
            assert app.blend_mode.get() == '滤色' and 'global' in forbidden, forbidden
            renderer = LivePreview(app)
            observed = []
            def task(value):
                time.sleep(.06)
                return value
            renderer.submit('same-photo', lambda: task(1), observed.append, lambda exc: (_ for _ in ()).throw(exc))
            pump(app, .02)
            renderer.submit('same-photo', lambda: task(2), observed.append, lambda exc: (_ for _ in ()).throw(exc))
            pump(app, .09)
            assert observed == [1], observed
            renderer.submit('new-photo', lambda: task(3), observed.append, lambda exc: (_ for _ in ()).throw(exc))
            pump(app, .18)
            assert observed == [1, 3], observed
            return {'slider_events_seconds': events, 'drag_values': exposure_values,
                    'final_ev': app.base_exposure_tenths.get()/10,
                    'delayed_frame_matches': True, 'no_global_rebuild': True,
                    'blend_selection_requests_new_frame': True,
                    'intermediate_and_new_source_order': observed}
        finally:
            app.destroy()


if __name__ == '__main__':
    print(json.dumps(run()))
