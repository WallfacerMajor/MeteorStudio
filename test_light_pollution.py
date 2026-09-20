import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from white_balance import encode_srgb, decode_srgb
from light_pollution import estimate, correct, export_image, validate


class Token:
    cancelled = False
    def raise_if_cancelled(self):
        if self.cancelled:
            raise RuntimeError('cancelled')


def fixture():
    y = np.linspace(0, 1, 240)[:, None, None]
    linear = np.broadcast_to(.02+y*np.array([.07, .04, .015]), (240, 360, 3)).copy()
    linear[80:83, 160:163] += .2
    return np.rint(encode_srgb(linear)*65535).astype(np.uint16)


def settings(**changes):
    return dict(dict(direction='bottom', start=0, falloff=1, strength=1, protected=[]), **changes)


class LightPollutionTests(unittest.TestCase):
    def test_real_tk_workflow(self):
        import gc
        import tkinter as tk
        from light_pollution_workspace import LightPollutionWindow
        from light_pollution_smoke import exercise_light_pollution
        from toolbox_smoke import pump
        root = tk.Tk()
        root.withdraw()
        window = LightPollutionWindow(root)
        try:
            exercise_light_pollution(root, window)
        finally:
            window._request_close()
            pump(root, 1.4)
            root.destroy()
            del root, window
            gc.collect()

    def test_gradient_recovery_and_star_signal(self):
        image = fixture()
        model = estimate(image, settings(), Token())
        np.testing.assert_allclose(model['amplitude'], [.07, .04, .015], atol=.002)
        output = decode_srgb(correct(image, settings(), model).astype(float)/65535)
        self.assertLess(float(np.max(abs(output[220, 20]-output[20, 20]))), .003)
        self.assertGreater(float(output[81,161,0]-output[81,140,0]), .19)

    def test_protection_and_zero_are_exact(self):
        image = fixture()
        model = estimate(image, settings(), Token())
        np.testing.assert_array_equal(correct(image, settings(strength=0), model), image)
        output = correct(image, settings(protected=[[0,.7,1,1]]), model)
        np.testing.assert_array_equal(output[170:], image[170:])
        self.assertTrue(np.any(output[100] != image[100]))

    def test_uniform_background_no_forced_correction(self):
        image = np.full((100, 140, 3), [16000, 12000, 10000], dtype=np.uint16)
        model = estimate(image, settings(), Token())
        self.assertEqual(model['amplitude'], [0, 0, 0])
        np.testing.assert_array_equal(correct(image, settings(), model), image)

    def test_direction_and_insufficient_samples(self):
        image = fixture()[::-1].copy()
        model = estimate(image, settings(direction='top'), Token())
        self.assertGreater(model['amplitude'][0], .06)
        with self.assertRaises(ValueError):
            estimate(image, settings(protected=[[0,0,1,1]]), Token())
        with self.assertRaises(ValueError):
            validate(settings(strength=float('nan')))
        token = Token()
        token.cancelled = True
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            estimate(image, settings(), token)

    def test_export_precision_and_readonly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'/'image.tif'
            source.parent.mkdir()
            pixels = fixture()
            tifffile.imwrite(source, pixels, photometric='rgb')
            before = source.read_bytes()
            model = estimate(pixels, settings(), Token())
            out = export_image(source, root/'out', settings(), model, Token(), lambda *_: None)
            np.testing.assert_array_equal(tifffile.imread(out/'result.tif'), correct(pixels, settings(), model))
            self.assertEqual(before, source.read_bytes())
            with self.assertRaises(ValueError):
                export_image(source, source.parent, settings(), model, Token(), lambda *_: None)

    def test_viewport_matches_full_resolution(self):
        from white_balance import make_pyramid, render_view
        pixels = fixture()
        model = estimate(pixels, settings(), Token())
        full = correct(pixels, settings(), model)
        result, position, clipped = render_view(make_pyramid(pixels, Token()), {}, 1., (180,120), (100,80),
            processor=lambda view,x,y,z: correct(view, settings(), model, pixels.shape[:2], (x,y), 1/z))
        expected = ((full[80:160,130:230].astype(np.uint32)+128)//257).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_cancelled_export_has_no_completed_file(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'/'image.tif'
            source.parent.mkdir()
            pixels = fixture()
            tifffile.imwrite(source, pixels, photometric='rgb')
            model = estimate(pixels, settings(), Token())
            token = Token()
            def progress(*args):
                token.cancelled = True
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                export_image(source, root/'out', settings(), model, token, progress)
            folder = next((root/'out').iterdir())
            self.assertFalse((folder/'result.tif').exists())
            self.assertEqual(json.loads((folder/'light_pollution.json').read_text(encoding='utf-8'))['status'], 'cancelled')
