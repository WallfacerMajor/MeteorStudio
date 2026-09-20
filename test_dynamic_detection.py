import unittest
from unittest.mock import patch

import cv2
import numpy as np

from meteor_composer import detect_trails


class DynamicDetectionTests(unittest.TestCase):
    def test_default_uses_material_geometry_not_composite_base(self):
        source = np.zeros((360, 560, 3), np.uint8)
        base = source.copy()
        region = np.ones(source.shape[:2], np.uint8) * 255
        with patch('meteor_composer.estimate_star_sky_mask', return_value=region) as estimate:
            detect_trails(source, base, ranked=True)
        self.assertIs(estimate.call_args.args[0], source)
        self.assertIs(estimate.call_args.args[1], source)

    def test_low_sky_trail_survives_but_ground_stays_excluded(self):
        base = np.random.default_rng(6021).integers(10, 30, (500, 800, 3), dtype=np.uint8)
        source = base.copy()
        cv2.line(source, (130, 400), (135, 450), (255, 240, 220), 2, cv2.LINE_AA)
        cv2.line(source, (300, 480), (600, 490), (255, 255, 255), 3, cv2.LINE_AA)
        region = np.zeros(source.shape[:2], np.uint8)
        region[:465] = 255
        with patch('meteor_composer.estimate_star_sky_mask', return_value=region):
            trails, _ = detect_trails(source, base, ranked=True)
        self.assertTrue(any(110 < (a[0] + b[0]) / 2 < 155 for a, b, _ in trails))
        self.assertFalse(any((a[1] + b[1]) / 2 > 465 for a, b, _ in trails))

    def test_screening_temporal_mask_is_not_replaced(self):
        source = np.zeros((100, 160, 3), np.uint8)
        with patch('meteor_composer.estimate_star_sky_mask', side_effect=AssertionError('unexpected estimate')):
            detect_trails(source, source, ranked=True, valid_region=np.ones((100, 160), np.uint8))
