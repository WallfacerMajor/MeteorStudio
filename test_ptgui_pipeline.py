import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin

from ptgui_pipeline import (
    ImageLensInfo,
    _set_independent_lenses,
    filter_sky_stars,
    make_star_sky_mask,
    read_lens_info,
)


class LensMetadataTests(unittest.TestCase):
    def test_reads_focal_length_per_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zoomed.jpg"
            exif = Image.Exif()
            exif[37386] = TiffImagePlugin.IFDRational(35, 1)
            exif[41989] = 52
            Image.new("RGB", (64, 48), "black").save(path, exif=exif)
            info = read_lens_info(path, 14.0, 43.2666)
            self.assertAlmostEqual(info.focal_length, 35.0)
            self.assertAlmostEqual(info.sensor_diagonal, 43.2666 * 35.0 / 52.0, places=4)
            self.assertEqual(info.source, "EXIF")

    def test_missing_exif_uses_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plain.png"
            Image.new("RGB", (64, 48), "black").save(path)
            info = read_lens_info(path, 20.0, 28.4)
            self.assertEqual(info.focal_length, 20.0)
            self.assertEqual(info.sensor_diagonal, 28.4)
            self.assertIn("EXIF缺失", info.source)

    def test_ptgui_groups_keep_different_focal_lengths(self):
        lens_template = {
            "lens": {
                "params": {"projection": "rectilinear", "focallength": 14.0, "sensordiagonal": 43.2666},
                "optimizerflags": {"fov": False, "a": False, "b": False, "c": False, "fisheyefactor": False},
            },
            "shift": {"params": {}, "optimizerflags": {"longside": False, "shortside": False}},
            "shear": {"params": {}, "optimizerflags": {"hshear": False, "vshear": False}},
        }
        project = {
            "globallenses": [lens_template],
            "imagegroups": [{"globallens": 0}, {"globallens": 0}],
            "panoramaparams": {},
            "outputsize": {},
        }
        infos = [
            ImageLensInfo(14.0, 43.2666, 14.0, "EXIF"),
            ImageLensInfo(24.0, 43.2666, 24.0, "EXIF"),
        ]
        _set_independent_lenses(project, infos, 6000, 4000, optimize_distortion=True)
        self.assertEqual([group["globallens"] for group in project["imagegroups"]], [0, 1])
        self.assertEqual([lens["lens"]["params"]["focallength"] for lens in project["globallenses"]], [14.0, 24.0])
        self.assertFalse(project["globallenses"][0]["lens"]["optimizerflags"]["a"])
        self.assertTrue(project["globallenses"][1]["lens"]["optimizerflags"]["a"])


class AutomaticSkyMaskTests(unittest.TestCase):
    def test_dense_sky_field_rejects_isolated_ground_points(self):
        sky = np.array(
            [(x, y) for y in range(40, 390, 45) for x in range(40, 760, 55)],
            dtype=np.float32,
        )
        ground = np.array([(70, 570), (230, 540), (430, 585), (650, 550)], dtype=np.float32)
        filtered = filter_sky_stars(np.vstack([sky, ground]), (600, 800))
        self.assertGreaterEqual(len(filtered), len(sky) * 0.8)
        self.assertLess(float(filtered[:, 1].max()), 500.0)
        mask = make_star_sky_mask((600, 800), filtered, radius=12)
        self.assertGreater(np.count_nonzero(mask[:450]), 0)
        self.assertEqual(np.count_nonzero(mask[520:]), 0)


if __name__ == "__main__":
    unittest.main()
