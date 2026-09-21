import gc
import json
import tempfile
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from background_tasks import CancellationToken, TaskCancelledError
from white_balance import make_lut, apply_lut, sample_neutral, validate_settings, export_image, read_source
from white_balance import _write_export_record, export_batch


class WhiteBalanceTests(unittest.TestCase):
    def test_record_retries_windows_errors_without_truncating_old_record(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "batch.json"
                path.write_text('{"status": "running"}', encoding="utf-8")
                original_replace = Path.replace
                attempts = []
                def replace(temporary, target):
                    attempts.append(temporary)
                    self.assertEqual(json.loads(path.read_text())["status"], "running")
                    if len(attempts) < 3:
                        error = PermissionError("record held open")
                        error.winerror = code
                        raise error
                    return original_replace(temporary, target)
                with patch.object(Path, "replace", replace), patch("white_balance.time.sleep") as sleep:
                    _write_export_record(path, {"status": "complete"})
                self.assertEqual(len(attempts), 3)
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(json.loads(path.read_text())["status"], "complete")
                self.assertFalse(path.with_name("batch.json.tmp").exists())

    def test_persistent_record_error_is_bounded_and_keeps_recovery_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.json"
            path.write_text('{"status": "running"}', encoding="utf-8")
            error = PermissionError("record remains locked")
            error.winerror = 5
            with patch.object(Path, "replace", side_effect=error) as replace, patch("white_balance.time.sleep") as sleep:
                with self.assertRaises(PermissionError) as caught:
                    _write_export_record(path, {"status": "complete"})
            self.assertIs(caught.exception, error)
            self.assertEqual(replace.call_count, 7)
            self.assertAlmostEqual(sum(call.args[0] for call in sleep.call_args_list), 2)
            self.assertEqual(json.loads(path.read_text())["status"], "running")
            self.assertEqual(json.loads(path.with_name("batch.json.tmp").read_text())["status"], "complete")

    def test_unrelated_record_errors_are_not_retried(self):
        for error in (PermissionError(13, "permanent permission"), OSError(28, "disk full")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                with patch.object(Path, "replace", side_effect=error) as replace, patch("white_balance.time.sleep") as sleep:
                    with self.assertRaises(OSError):
                        _write_export_record(Path(folder) / "batch.json", {})
                replace.assert_called_once()
                sleep.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows file sharing semantics")
    def test_batch_and_image_records_survive_real_windows_locks(self):
        for record_name in ("batch.json", "white_balance.json"):
            with self.subTest(record=record_name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "source").mkdir()
                source = root / "source" / "night.tif"
                pixels = np.full((96, 128, 3), 23456, np.uint16)
                tifffile.imwrite(source, pixels, photometric="rgb")
                original = source.read_bytes()
                timers = []
                def progress(*_):
                    if not timers:
                        path = next((root / "out").rglob(record_name), None)
                        if path is not None:
                            locked = path.open("rb")
                            timer = threading.Timer(.35, locked.close)
                            timers.append(timer)
                            timer.start()
                try:
                    output = export_batch([source], root / "out", {}, CancellationToken("test", 1), progress)
                finally:
                    for timer in timers:
                        timer.join()
                self.assertTrue(timers, "Test did not hold the record open")
                report = json.loads((output / "batch.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "complete")
                item = Path(report["items"][0]["output"])
                self.assertEqual(json.loads((item / "white_balance.json").read_text(encoding="utf-8"))["status"], "complete")
                np.testing.assert_array_equal(tifffile.imread(item / "result.tif"), pixels)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(list(output.rglob("*.tmp")), [])

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
