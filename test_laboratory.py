import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tifffile
from background_tasks import CancellationToken, TaskCancelledError
from laboratory import run_experiment, quality_metrics


class LaboratoryTests(unittest.TestCase):
    def test_png_preserves_16_bit_channels(self):
        import cv2
        from laboratory import read_pixels
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "image.png"
            rgb = np.full((12, 15, 3), [12345, 23456, 34567], np.uint16)
            cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tofile(path)
            np.testing.assert_array_equal(read_pixels(path), rgb)

    def test_streaming_results_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            arrays = [np.full((30, 40, 3), v, np.uint16) for v in (1001, 2003, 3005)]
            paths = [source / f"{i}.tif" for i in range(3)]
            for p, a in zip(paths, arrays):
                tifffile.imwrite(p, a, photometric="rgb")
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            for mode, value in (("trails", 3005), ("mean", 2003)):
                out = run_experiment(paths, root / "output", mode, CancellationToken("test", 1))
                result = tifffile.imread(out / "result.tif")
                self.assertEqual(result.dtype, np.uint16)
                np.testing.assert_array_equal(result, np.full_like(arrays[0], value))
            out = run_experiment(paths, root / "output", "quality", CancellationToken("test", 1))
            self.assertTrue((out / "quality.csv").is_file())
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            with self.assertRaises(ValueError):
                run_experiment(paths, source, "mean", CancellationToken("test", 1))

    def test_cancel_marks_incomplete_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            paths = [source / f"{i}.tif" for i in range(2)]
            for p in paths:
                tifffile.imwrite(p, np.zeros((10, 10, 3), np.uint16), photometric="rgb")
            token = CancellationToken("test", 1)
            with self.assertRaises(TaskCancelledError):
                run_experiment(paths, root / "out", "mean", token, lambda *_: token.cancel())
            out = next((root / "out").iterdir())
            self.assertFalse((out / "result.tif").exists())
            self.assertEqual(json.loads((out / "experiment.json").read_text(encoding="utf-8"))["status"], "cancelled")

    def test_metrics_measure_clipping(self):
        image = np.zeros((20, 20, 3), np.uint16)
        image[:10] = 65535
        self.assertEqual(quality_metrics(image)["clipped_fraction"], 0.5)

    def test_real_tk_lab(self):
        import tkinter as tk
        from unittest.mock import patch
        from laboratory_workspace import LaboratoryWindow
        from toolbox_smoke import click, pump
        with tempfile.TemporaryDirectory() as folder:
            root_path = Path(folder)
            source = root_path / "source"
            source.mkdir()
            paths = [source / f"{i}.tif" for i in range(2)]
            for p in paths:
                tifffile.imwrite(p, np.full((64, 80, 3), 12345, np.uint16), photometric="rgb")
            root = tk.Tk()
            root.withdraw()
            window = LaboratoryWindow(root, "mean")
            try:
                with patch("laboratory_workspace.filedialog.askopenfilenames", return_value=tuple(map(str, paths))):
                    click(root, window.add_button)
                window.destination.set(str(root_path / "out"))
                click(root, window.start_button)
                pump(root, 1.6)
                self.assertFalse(window.busy)
                self.assertIsNotNone(window.result)
                self.assertTrue((window.result / "result.tif").exists())
                np.testing.assert_array_equal(tifffile.imread(window.result / "result.tif"), np.full((64,80,3),12345,np.uint16))
            finally:
                window._request_close()
                pump(root, 1.4)
                root.destroy()
                import gc
                del window, root
                gc.collect()
