import unittest

import numpy as np

import meteor_composer
from meteor_detection import (
    ML_FEATURE_NAMES,
    calibrate_secondary_candidate_scores,
    candidate_feature_vector,
    prepare_ml_maps,
)


class MeteorDetectionModuleTests(unittest.TestCase):
    def test_composer_reexports_shared_scoring_functions(self):
        self.assertIs(meteor_composer.prepare_ml_maps, prepare_ml_maps)
        self.assertIs(meteor_composer.candidate_feature_vector, candidate_feature_vector)
        self.assertIs(
            meteor_composer.calibrate_secondary_candidate_scores,
            calibrate_secondary_candidate_scores,
        )

    def test_feature_vector_shape_and_values_are_stable(self):
        base = np.full((96, 128, 3), 20, dtype=np.uint8)
        source = base.copy()
        for offset in range(35):
            source[30 + offset // 4, 35 + offset] = (160, 175, 210)
        features = candidate_feature_vector(
            prepare_ml_maps(source, base), (35, 30), (69, 38), 82.0
        )
        self.assertEqual(features.shape, (len(ML_FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(float(features[0]), 0.82, places=5)

    def test_secondary_calibration_keeps_weak_candidate_available(self):
        candidates = [
            (88, (10, 20), (170, 20), 96.0),
            (91, (30, 40), (65, 40), 40.0),
        ]
        calibrated = calibrate_secondary_candidate_scores(candidates)
        self.assertEqual(calibrated[0][0], 88)
        self.assertEqual(calibrated[1][0], 49)


if __name__ == "__main__":
    unittest.main()
