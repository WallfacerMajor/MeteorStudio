import unittest
from types import SimpleNamespace

import numpy as np

from meteor_composer import MeteorComposer, normalize_lsd_lines, point_in_padded_bbox


class NormalizeLsdLinesTests(unittest.TestCase):
    def test_accepts_opencv_nested_shape(self):
        lines = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.float32)
        result = normalize_lsd_lines(lines)
        self.assertEqual(result.shape, (2, 4))
        np.testing.assert_array_equal(result[1], [5, 6, 7, 8])

    def test_accepts_flat_line_shape(self):
        lines = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        result = normalize_lsd_lines(lines)
        self.assertEqual(result.shape, (2, 4))

    def test_accepts_single_flat_line(self):
        result = normalize_lsd_lines(np.array([1, 2, 3, 4], dtype=np.float32))
        self.assertEqual(result.shape, (1, 4))

    def test_none_is_empty(self):
        self.assertEqual(normalize_lsd_lines(None).shape, (0, 4))


class CandidateButtonGeometryTests(unittest.TestCase):
    def test_button_hit_area(self):
        bbox = (100, 50, 200, 80)
        self.assertTrue(point_in_padded_bbox(150, 65, bbox))
        self.assertFalse(point_in_padded_bbox(90, 65, bbox))

    def test_padding_bridges_pointer_gap(self):
        bbox = (100, 50, 200, 80)
        self.assertTrue(point_in_padded_bbox(90, 65, bbox, padding=16))
        self.assertFalse(point_in_padded_bbox(80, 65, bbox, padding=16))

    def test_candidate_button_press_does_not_start_brush(self):
        picked = []
        fake = SimpleNamespace(
            current_path="image.tif",
            candidate_click_active=False,
            hover_candidate_items=[1, 2],
            canvas=SimpleNamespace(bbox=lambda _tag: (100, 50, 200, 80)),
            _pick_hover_candidate=lambda event: picked.append(event),
        )
        event = SimpleNamespace(x=150, y=65)
        result = MeteorComposer._stroke_start(fake, event)
        self.assertEqual(result, "break")
        self.assertEqual(picked, [event])

    def test_candidate_release_clears_click_without_creating_stroke(self):
        fake = SimpleNamespace(
            candidate_click_active=True,
            active_points=[(0.5, 0.5)],
            active_canvas_line=123,
        )
        result = MeteorComposer._stroke_end(fake, SimpleNamespace())
        self.assertEqual(result, "break")
        self.assertFalse(fake.candidate_click_active)
        self.assertEqual(fake.active_points, [])
        self.assertIsNone(fake.active_canvas_line)


if __name__ == "__main__":
    unittest.main()
