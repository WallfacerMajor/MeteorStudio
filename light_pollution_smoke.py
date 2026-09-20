"""Real pointer workflow shared by source tests and the packaged executable."""
import hashlib
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from white_balance import encode_srgb
from light_pollution import correct


def exercise_light_pollution(root, window):
    from toolbox_smoke import click, pump, reveal_control, capture
    def wait_for(predicate):
        deadline = time.monotonic()+15
        while not predicate() and time.monotonic() < deadline:
            pump(root, .04)
        assert predicate(), window.status.get()
        pump(root, 1.45)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        source = folder/'source'/'night.tif'
        source.parent.mkdir()
        linear = np.broadcast_to(.02+np.linspace(0,1,600)[:,None,None]*np.array([.07,.04,.015]), (600,900,3)).copy()
        linear[220:224,440:444] += .3
        linear[510:] = [.01,.025,.009]
        pixels = np.rint(encode_srgb(linear)*65535).astype(np.uint16)
        tifffile.imwrite(source, pixels, photometric='rgb')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with patch('light_pollution_workspace.filedialog.askopenfilename', return_value=str(source)):
            click(root, window.open_button)
        wait_for(lambda: window.levels is not None and window.photo is not None and not window.busy)
        assert window.edit_inspector.winfo_rootx() >= window.canvas.winfo_rootx()+window.canvas.winfo_width()
        click(root, window.protect_button)
        a,b = window.point(0,.81)
        c,d = window.point(.999,.999)
        window.canvas.event_generate('<ButtonPress-1>', x=round(a), y=round(b))
        window.canvas.event_generate('<B1-Motion>', x=round(c), y=round(d))
        window.canvas.event_generate('<ButtonRelease-1>', x=round(c), y=round(d))
        pump(root, .2)
        assert len(window.protected) == 1
        click(root, window.protect_button)
        click(root, window.analyze_button)
        wait_for(lambda: window.model is not None and not window.busy)
        assert window.model['amplitude'][0] > .05
        view = (window.zoom, window.center, window.canvas.bbox(window.image_item), window.canvas.winfo_width(), window.canvas.winfo_height())
        click(root, window.compare_button)
        pump(root, 1.45)
        assert '原图' in window.info.get()
        click(root, window.compare_button)
        scale = window.sliders[2]
        reveal_control(root, scale)
        thumb_x, thumb_y = scale.coords(window.strength.get())
        scale.event_generate('<Enter>', x=round(thumb_x), y=round(thumb_y))
        scale.event_generate('<ButtonPress-1>', x=round(thumb_x), y=round(thumb_y))
        scale.event_generate('<B1-Motion>', x=round(scale.winfo_width()*.82), y=scale.winfo_height()//2)
        scale.event_generate('<ButtonRelease-1>', x=round(scale.winfo_width()*.82), y=scale.winfo_height()//2)
        pump(root, 1.45)
        assert window.strength.get() > 90
        assert view == (window.zoom, window.center, window.canvas.bbox(window.image_item), window.canvas.winfo_width(), window.canvas.winfo_height())
        expected = correct(pixels, window.pollution_settings(), window.model)
        window.destination.set(str(folder/'out'))
        click(root, window.export_button)
        wait_for(lambda: window.result is not None and not window.busy)
        actual = tifffile.imread(window.result/'result.tif')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[520:580,20:880], pixels[520:580,20:880])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        previous = window.result
        def slow_export(source, destination, settings, model, token, progress):
            for _ in range(100):
                token.raise_if_cancelled()
                time.sleep(.02)
            raise AssertionError('Cancel was not delivered')
        with patch('light_pollution_workspace.export_image', side_effect=slow_export):
            click(root, window.export_button)
            click(root, window.cancel_button)
            wait_for(lambda: not window.busy)
        assert window.result == previous
        model = window.model
        window.events.put(('modeled', window.generation-1, {'amplitude':[.5]*3, 'samples':99}))
        pump(root, 1.45)
        assert window.model is model
        capture(window, 'light-pollution.png')
        window.geometry('850x600')
        pump(root, 1.45)
        reveal_control(root, window.export_button)
        assert window.canvas.winfo_height() > 200
        assert window.edit_inspector.winfo_rootx() >= window.canvas.winfo_rootx()+window.canvas.winfo_width()
        capture(window, 'light-pollution-small.png')
        click(root, window.clear_button)
        pump(root, 1.45)
        assert not window.protected and window.model is None and str(window.export_button['state']) == 'disabled'
    return {'lp_gradient_preview':'passed', 'lp_protected_pixels':'passed', 'lp_16bit_export':'passed', 'lp_cancel_stale_jobs':'passed', 'lp_right_small_window':'passed'}
