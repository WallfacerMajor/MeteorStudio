import unittest
from unittest.mock import patch
import numpy as np
from meteor_composer import Stroke, transformed_object_crop, remove_local_background_cast


class CandidateBoundsTests(unittest.TestCase):
    def test_background_fit_factors_once_and_matches_separate_channels(self):
        rng = np.random.default_rng(42)
        source = rng.uniform(0, 255, (75, 100, 3)).astype(np.float32)
        base = rng.uniform(0, 255, source.shape).astype(np.float32)
        alpha = np.zeros(source.shape[:2], np.float32)
        alpha[30:40, 20:80] = 1
        solve = np.linalg.lstsq

        def separate_channels(design, samples, rcond=None):
            return (np.column_stack([
                solve(design, samples[:, channel], rcond=rcond)[0]
                for channel in range(3)
            ]),)

        with patch('meteor_composer.np.linalg.lstsq', side_effect=separate_channels):
            expected = remove_local_background_cast(source, base, alpha, 70)
        with patch('meteor_composer.np.linalg.lstsq', wraps=solve) as solver:
            actual = remove_local_background_cast(source, base, alpha, 70)
        self.assertEqual(solver.call_count, 1)
        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.00001)

    def test_geometry_only_matches_real_crop_without_raster_work(self):
        image = np.zeros((480, 720, 3), np.uint8)
        for rotation in (0, 37, -85):
            for scale in (0.3, 1, 3):
                stroke = Stroke([(0.12, 0.25), (0.7, 0.6)], 17, 23)
                stroke.rotation = rotation
                stroke.length_scale = scale
                stroke.offset_x = 25
                expected = transformed_object_crop(image, stroke)
                with patch('meteor_composer.build_mask_crop', side_effect=AssertionError('raster mask')):
                    with patch('meteor_composer.cv2.warpAffine', side_effect=AssertionError('pixel warp')):
                        actual = transformed_object_crop(image, stroke, bounds_only=True)
                self.assertEqual(actual[3], expected[3])
                self.assertEqual(actual[:3], (None, None, None))
