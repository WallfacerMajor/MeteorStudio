import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from meteor_screening import (
    IMAGE_SUFFIXES, RAW_SUFFIXES, MeteorScreeningWindow, ScreeningCandidate, ScreeningResult,
    mark_temporal_repeats,
    estimate_star_sky_mask,
    temporal_reference,
)
from meteor_composer import detect_trails
from meteor_learning import build_screening_feedback_dataset


class TemporalReferenceTests(unittest.TestCase):
    def test_common_sony_nikon_canon_raw_formats_are_accepted(self):
        expected = {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw"}
        self.assertTrue(expected.issubset(RAW_SUFFIXES))
        self.assertTrue(expected.issubset(IMAGE_SUFFIXES))

    def test_neighbor_median_excludes_current_frame_meteor(self):
        base = np.full((180, 280, 3), 12, dtype=np.uint8)
        for x, y in ((30, 25), (80, 42), (140, 65), (210, 38), (245, 92), (55, 115)):
            cv2.circle(base, (x, y), 2, (190, 190, 190), -1, cv2.LINE_AA)
        current = base.copy()
        cv2.line(current, (48, 130), (230, 82), (245, 248, 255), 4, cv2.LINE_AA)
        reference, displacement = temporal_reference(current, [base.copy() for _ in range(4)])
        self.assertLess(displacement, 1.2)
        self.assertLess(int(reference[108, 135].max()), 40)
        self.assertGreater(int(current[108, 135].max()), 180)

    def test_dynamic_star_sky_mask_excludes_high_uneven_landscape(self):
        rng = np.random.default_rng(8)
        height, width = 360, 560
        reference = np.full((height, width, 3), 12, dtype=np.uint8)
        horizon = np.asarray([
            220 + 42 * np.sin(x / 70.0) + (32 if 215 < x < 320 else 0)
            for x in range(width)
        ], dtype=int)
        for x, bottom in enumerate(horizon):
            texture = rng.integers(0, 55, size=(height - bottom, 3), dtype=np.uint8)
            reference[bottom:, x] = 34 + texture
        for _ in range(520):
            x = int(rng.integers(5, width - 5))
            y = int(rng.integers(5, max(6, horizon[x] - 5)))
            value = int(rng.integers(145, 245))
            cv2.circle(reference, (x, y), 1, (value, value, value), -1, cv2.LINE_AA)
        current = reference.copy()
        cv2.line(current, (70, 90), (230, 125), (240, 245, 255), 3, cv2.LINE_AA)
        cv2.line(current, (340, 300), (520, 325), (245, 245, 245), 3, cv2.LINE_AA)
        mask = estimate_star_sky_mask(current, reference)
        self.assertGreater(float(np.mean(mask[60:150] > 0)), 0.92)
        self.assertLess(float(np.mean(mask[310:350] > 0)), 0.10)
        trails, _ = detect_trails(current, reference, ranked=True, valid_region=mask)
        self.assertTrue(any((start[1] + end[1]) / 2 < 180 for start, end, _score in trails))
        self.assertFalse(any((start[1] + end[1]) / 2 > 280 for start, end, _score in trails))

    def test_temporal_repeat_marks_probable_aircraft_sequence(self):
        results = []
        for index in range(3):
            candidate = ScreeningCandidate(
                (20 + index * 25, 60), (100 + index * 25, 70), 82
            )
            results.append(ScreeningResult(f"frame_{index}.tif", [candidate], 82))
        mark_temporal_repeats(results, (180, 300))
        self.assertGreaterEqual(results[1].temporal_hits, 2)
        self.assertLess(results[1].score, 82)
        self.assertIn("飞机/卫星", results[1].note)

    def test_candidate_guides_do_not_cover_meteor_center(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        candidate = ScreeningCandidate((30, 60), (210, 60), 88)
        MeteorScreeningWindow._draw_candidate_marker(image, candidate, (255, 190, 45), True)
        self.assertTrue(np.all(image[60, 120] == 0))
        self.assertGreater(int(image[53, 120].max()), 0)

    def test_learning_uses_only_explicit_candidate_labels(self):
        feature_names = [f"f{index}" for index in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feedback.json"
            path.write_text(json.dumps([
                {"id": "exact", "source_path": "a.arw", "label": 1,
                 "features": [1, 2, 3, 4], "legacy": 0.8},
                {"id": "image-only", "source_path": "b.arw", "decision": "accept",
                 "features": [4, 3, 2, 1]},
            ]), encoding="utf-8")
            data = build_screening_feedback_dataset({
                "ML_FEATURE_NAMES": feature_names,
                "screening_feedback_path": path,
            })
        self.assertEqual(data["x"].shape, (1, 4))
        self.assertEqual(data["y"].tolist(), [1])

    def test_candidate_labels_drive_photo_decision_only_when_conclusive(self):
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window.decisions = {}
        window.decision_sources = {}
        first = ScreeningCandidate((10, 10), (80, 40), 70, label="not_meteor")
        second = ScreeningCandidate((20, 70), (100, 50), 65)
        result = ScreeningResult("frame.arw", [first, second], 70)
        window._sync_photo_decision_from_candidates(result)
        self.assertNotIn(result.path, window.decisions)
        second.label = "not_meteor"
        window._sync_photo_decision_from_candidates(result)
        self.assertEqual(window.decisions[result.path], "reject")
        self.assertEqual(window.decision_sources[result.path], "candidate")
        second.label = ""
        window._sync_photo_decision_from_candidates(result)
        self.assertNotIn(result.path, window.decisions)
        second.label = "meteor"
        window._sync_photo_decision_from_candidates(result)
        self.assertEqual(window.decisions[result.path], "accept")


if __name__ == "__main__":
    unittest.main()
