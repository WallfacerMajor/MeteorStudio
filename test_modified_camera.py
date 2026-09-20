import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import tifffile
from background_tasks import CancellationToken, TaskCancelledError
from white_balance import make_lut, validate_settings, read_source, export_batch


class ModifiedCameraTests(unittest.TestCase):
    def test_strength_zero_does_not_neutralize_red(self):
        lut = make_lut(dict(neutral=[.3, 1, 2], neutral_strength=0))
        for channel in range(3):
            np.testing.assert_array_equal(lut[:, channel], np.arange(65536))
        # Equipment names are records, never fixed red suppression presets.
        np.testing.assert_array_equal(make_lut({"equipment": {"modification": "Hα 增强"}}), make_lut({}))

    def test_fixed_daylight_never_uses_per_frame_camera_wb(self):
        raw = MagicMock()
        raw.postprocess.return_value = np.full((4, 6, 3), 1000, np.uint16)
        raw.__enter__.return_value = raw
        with patch("rawpy.imread", return_value=raw):
            read_source(Path("test.arw"), "daylight")
        self.assertFalse(raw.postprocess.call_args.kwargs["use_camera_wb"])
        self.assertFalse(raw.postprocess.call_args.kwargs["use_auto_wb"])
        self.assertTrue(raw.postprocess.call_args.kwargs["no_auto_bright"])

    def test_preset_validation(self):
        data = validate_settings(dict(raw_baseline="daylight", neutral_strength=35, equipment={"camera": "A", "filter": "B"}))
        self.assertEqual(data["equipment"]["camera"], "A")
        for invalid in ({"neutral_strength": float("nan")}, {"raw_baseline": "automatic"}, {"equipment": []}):
            with self.assertRaises(ValueError):
                validate_settings(invalid)

    def test_batch_failure_is_reported_and_other_images_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            a, b = source / "a.tif", source / "broken.tif"
            tifffile.imwrite(a, np.full((16, 20, 3), 20000, np.uint16), photometric="rgb")
            b.write_bytes(b"invalid")
            result = export_batch([a, b, a], root / "out", {}, CancellationToken("test", 1))
            report = json.loads((result / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete_with_errors")
            self.assertEqual([item["status"] for item in report["items"]], ["complete", "failed"])
            self.assertTrue((Path(report["items"][0]["output"]) / "result.tif").exists())

    def test_batch_cancel_reports_remaining_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            paths = [source / f"{i}.tif" for i in range(2)]
            for path in paths:
                tifffile.imwrite(path, np.full((300, 20, 3), 20000, np.uint16), photometric="rgb")
            token = CancellationToken("test", 1)
            def progress(value, message):
                if value > 0:
                    token.cancel()
            with self.assertRaises(TaskCancelledError):
                export_batch(paths, root / "out", {}, token, progress)
            result = next((root / "out").iterdir())
            report = json.loads((result / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "cancelled")
            self.assertEqual(len(report["sources"]), 2)
            self.assertEqual(len(report["items"]), 1)
