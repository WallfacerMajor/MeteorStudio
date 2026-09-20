import tempfile
import json
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, TiffImagePlugin

from ptgui_pipeline import (
    ImageLensInfo,
    _set_independent_lenses,
    _run_cancellable_process,
    alignment_solution_quality,
    configure_layer_export,
    filter_sky_stars,
    make_star_sky_mask,
    load_confirmed_meteor_tracks,
    project_confirmed_tracks,
    ptgui_solution_sanity,
    read_lens_info,
    siril_find_stars,
    match_star_pairs,
    _unique_star_pairs,
)
from background_tasks import TaskCancelledError


class ExternalProcessCancellationTests(unittest.TestCase):
    def test_siril_success_returns_cancellable_runner_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proxy = root / "proxy.png"
            Image.new("RGB", (100, 80)).save(proxy)
            rows = []
            for x, y in ((10, 20), (30, 40), (50, 60)):
                columns = ['0'] * 16
                columns[5:9] = [str(x), str(y), '2', '2']
                # Siril can fit clipped bright stars; those remain usable.
                columns[14] = '1' if x == 30 else '0'
                rows.append(' '.join(columns))
            (root / 'case_stars.lst').write_text('\n'.join(rows), encoding='utf-8')
            with patch('ptgui_pipeline._run_cancellable_process', return_value=(0, 'Siril success log')):
                stars, log = siril_find_stars(Path('siril'), proxy, root, 'case')
            np.testing.assert_array_equal(stars, [[10, 20], [30, 40], [50, 60]])
            self.assertEqual(stars.dtype, np.float32)
            self.assertEqual(log, 'Siril success log')

    def test_siril_failure_preserves_process_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('ptgui_pipeline._run_cancellable_process', return_value=(1, 'specific failure')):
                with self.assertRaisesRegex(RuntimeError, 'specific failure'):
                    siril_find_stars(Path('siril'), Path('proxy.png'), Path(folder), 'case')

    def test_screening_tracks_load_and_project_through_star_solution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "confirmed_meteor_tracks.json").write_text(json.dumps({
                "format": "meteor-confirmed-tracks-v1",
                "items": [{
                    "exported_name": "frame.arw",
                    "tracks": [{
                        "start_normalized": [0.1, 0.2],
                        "end_normalized": [0.8, 0.7], "score": 91,
                    }],
                }],
            }), encoding="utf-8")
            loaded = load_confirmed_meteor_tracks(root)
        pairs = [
            (np.array((0, 0), np.float32), np.array((10, 20), np.float32)),
            (np.array((99, 0), np.float32), np.array((109, 20), np.float32)),
            (np.array((99, 49), np.float32), np.array((109, 69), np.float32)),
            (np.array((0, 49), np.float32), np.array((10, 69), np.float32)),
        ]
        projected = project_confirmed_tracks(loaded["frame.arw"], pairs, 100, 50)
        self.assertEqual(len(projected), 1)
        self.assertAlmostEqual(projected[0]["aligned_start_normalized"][0], 0.1 + 10 / 99, places=4)
        self.assertAlmostEqual(projected[0]["aligned_start_normalized"][1], 0.2 + 20 / 49, places=4)

    def test_cancellation_terminates_running_process(self):
        checks = 0

        def cancel() -> None:
            nonlocal checks
            checks += 1
            if checks >= 2:
                raise TaskCancelledError("cancel test")

        with tempfile.TemporaryDirectory() as folder:
            started = time.monotonic()
            with self.assertRaises(TaskCancelledError):
                _run_cancellable_process(
                    [sys.executable, "-c", "import time; time.sleep(10)"],
                    Path(folder), cancel,
                )
        self.assertLess(time.monotonic() - started, 3.0)


class AlignmentSolutionQualityTests(unittest.TestCase):
    def test_sparse_psf_catalogue_does_not_discard_verified_correspondences(self):
        points = np.float32([(x, y) for y in (40, 160, 280) for x in (40, 160, 280, 400)])
        target = points + [7, 11]
        descriptors = np.eye(len(points), 128, dtype=np.float32)
        source_keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in points]
        target_keypoints = [cv2.KeyPoint(float(x), float(y), 3) for x, y in target]
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'stars.png'
            Image.new('L', (500, 400)).save(image)
            with patch('ptgui_pipeline.cv2.SIFT_create') as detector:
                detector.return_value.detectAndCompute.side_effect = [
                    (source_keypoints, descriptors), (target_keypoints, descriptors),
                ]
                pairs, error = match_star_pairs(image, image, points[:4], target[:4], .25)
        self.assertEqual(len(pairs), 12)
        self.assertLess(error, .01)
        for source, dest in pairs:
            np.testing.assert_allclose(dest-source, [28, 44], atol=.001)

    def test_multiple_sift_orientations_are_not_independent_control_points(self):
        source = np.float32([[.9, 2], [1.1, 2], [9, 8], [15, 20]])
        target = np.float32([[5, 6], [5.2, 6], [5.1, 6], [25, 30]])
        unique_source, unique_target = _unique_star_pairs(source, target)
        np.testing.assert_array_equal(unique_source, source[[0, 3]])
        np.testing.assert_array_equal(unique_target, target[[0, 3]])

    def test_five_control_points_are_rejected_as_unverified(self):
        accepted, review, message = alignment_solution_quality(5, 1.2)
        self.assertFalse(accepted)
        self.assertTrue(review)
        self.assertIn("5组", message)

    def test_three_control_points_are_not_enough_for_homography(self):
        accepted, review, message = alignment_solution_quality(3, 1.0)
        self.assertFalse(accepted)
        self.assertTrue(review)
        self.assertIn("至少需要6组", message)

    def test_six_accurate_control_points_are_normal(self):
        accepted, review, message = alignment_solution_quality(6, 1.0)
        self.assertTrue(accepted)
        self.assertFalse(review)
        self.assertEqual(message, "")

    def test_extreme_ptgui_roll_is_rejected(self):
        project = {
            "imagegroups": [
                {"position": {"params": {"yaw": 0, "pitch": 0, "roll": 0}}},
                {"position": {"params": {"yaw": -9, "pitch": 14, "roll": -59}}},
            ],
            "panoramaparams": {"hfov": 84, "vfov": 62},
            "globallenses": [],
        }
        accepted, message = ptgui_solution_sanity(project)
        self.assertFalse(accepted)
        self.assertIn("旋转", message)


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

    def test_lab_projection_changes_panorama_but_not_input_lens_type(self):
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
            "panoramaparams": {}, "outputsize": {},
        }
        infos = [ImageLensInfo(14.0, 43.2666, 14.0, "EXIF")] * 2
        _set_independent_lenses(
            project, infos, 6000, 4000,
            panorama_projection="mercator", canvas_scale=1.35,
        )
        self.assertEqual(project["panoramaparams"]["projection"], "mercator")
        self.assertGreater(project["panoramaparams"]["hfov"], 100.0)
        self.assertTrue(all(
            lens["lens"]["params"]["projection"] == "rectilinear"
            for lens in project["globallenses"]
        ))


class LayerExportConfigurationTests(unittest.TestCase):
    def test_following_projects_export_only_the_meteor_layer_without_compression(self):
        project = {
            "outputcomponents": {},
            "imagegroups": [
                {"images": [{"include": True, "includeinpreview": True}]},
                {"images": [{"include": True, "includeinpreview": True}]},
            ],
            "panoramaparams": {
                "tiffparams": {"compression": "deflate"},
                "outputcrop": [0.1, 0.1, 0.9, 0.9],
            },
            "projectsettings": {
                "alignsettings": {}, "batchstitchersettings": {},
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project_file = root / "single.pts"
            project_file.write_text(
                json.dumps({"project": project}), encoding="utf-8"
            )
            configure_layer_export(
                project_file, root / "aligned.tif", export_reference=False,
            )
            configured = json.loads(
                project_file.read_text(encoding="utf-8")
            )["project"]
        self.assertFalse(configured["imagegroups"][0]["images"][0]["include"])
        self.assertFalse(configured["imagegroups"][0]["images"][0]["includeinpreview"])
        self.assertTrue(configured["imagegroups"][1]["images"][0]["include"])
        self.assertEqual(configured["panoramaparams"]["tiffparams"]["datatype"], "u16")
        self.assertEqual(configured["panoramaparams"]["tiffparams"]["compression"], "none")
        self.assertFalse(configured["projectsettings"]["alignsettings"]["generatecp"])


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
