import unittest

import numpy as np

from meteor_composer import normalize_lsd_lines


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


if __name__ == "__main__":
    unittest.main()
