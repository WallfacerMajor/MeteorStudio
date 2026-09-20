import gc
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from background_tasks import CancellationToken, TaskCancelledError
from white_balance import make_lut, apply_lut, sample_neutral, validate_settings, export_image, read_source


class WhiteBalanceTests(unittest.TestCase):
    def test_identity_is_bit_exact(self):
        lut = make_lut({})
        for channel in range(3):
            np.testing.assert_array_equal(lut[:, channel], np.arange(65536))

    def test_controls_and_neutral(self):
        image = np.full((21, 21, 3), [24000, 18000, 12000], np.uint16)
        neutral = sample_neutral(image, 10, 10)
        pixel = apply_lut(image, make_lut({"neutral": neutral}))[10,10].astype(int)
        self.assertLessEqual(pixel.max()-pixel.min(), 2)
        warm = make_lut({"warmth": 70})[30000].astype(int)
        self.assertGreater(warm[0], warm[2])
        tint = make_lut({"tint": 70})[30000].astype(int)
        self.assertLess(tint[1], tint[0])
        with self.assertRaises(ValueError):
            sample_neutral(np.zeros_like(image), 10, 10)

    def test_reject_invalid_settings(self):
        for data in ([], {"warmth": float("nan")}, {"tint": 101}, {"neutral": [1,0,1]}, {"version": 2}):
            with self.assertRaises(ValueError):
                validate_settings(data)

    def test_cancel_does_not_publish_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            path = source / "image.tif"
            image = np.full((400, 30, 3), 23456, np.uint16)
            tifffile.imwrite(path, image, photometric="rgb")
            token = CancellationToken("test", 1)
            def progress(value, text):
                if value > 0:
                    token.cancel()
            with self.assertRaises(TaskCancelledError):
                export_image(path, Path(folder)/"out", {}, token, progress)
            output = next((Path(folder)/"out").iterdir())
            self.assertFalse((output/"result.tif").exists())
            self.assertEqual(json.loads((output/"white_balance.json").read_text(encoding="utf-8"))["status"], "cancelled")
            np.testing.assert_array_equal(tifffile.imread(path), image)
            with self.assertRaises(ValueError):
                export_image(path, source, {}, CancellationToken("test", 1))

    def test_export_icc_and_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"source"
            source.mkdir()
            path = source/"rotated.tif"
            image = np.arange(8*10*3, dtype=np.uint16).reshape(8,10,3)
            tifffile.imwrite(path, image, photometric="rgb", extratags=[(274,"H",1,6,False)])
            np.testing.assert_array_equal(read_source(path), np.rot90(image,-1))
            output = export_image(path, Path(folder)/"out", {}, CancellationToken("test",1))
            np.testing.assert_array_equal(read_source(output/"result.tif"), np.rot90(image,-1))
            with tifffile.TiffFile(output/"result.tif") as tif:
                self.assertIn(34675, tif.pages[0].tags)

    def test_real_tk_workflow(self):
        import tkinter as tk
        from white_balance_workspace import WhiteBalanceWindow
        from white_balance_smoke import exercise_white_balance
        from toolbox_smoke import pump
        root = tk.Tk()
        root.withdraw()
        window = WhiteBalanceWindow(root)
        try:
            exercise_white_balance(root, window)
        finally:
            window._request_close()
            pump(root, 1.4)
            root.destroy()
            del window, root
            gc.collect()
