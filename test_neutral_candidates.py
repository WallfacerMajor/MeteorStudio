import unittest
import numpy as np
from neutral_candidates import suggest_neutral_points


class Token:
    def raise_if_cancelled(self):
        pass


class NeutralCandidateTests(unittest.TestCase):
    def test_diverse_readonly_and_original_gains(self):
        image = np.full((400, 600, 3), [18000, 15000, 12000], dtype=np.uint16)
        image[100:150, 100:150] = 65535
        before = image.copy()
        points = suggest_neutral_points(image, Token())
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertFalse(100 <= p['x'] < 150 and 100 <= p['y'] < 150)
            self.assertLess(p['gains'][0], p['gains'][2])
        np.testing.assert_array_equal(image, before)

    def test_no_forced_reference_in_unsuitable_images(self):
        for value in ([0, 0, 0], [65535]*3, [50000, 5000, 5000]):
            self.assertEqual(suggest_neutral_points(np.full((200, 300, 3), value, dtype=np.uint16), Token()), [])

    def test_cancellation(self):
        class Cancel:
            def raise_if_cancelled(self):
                raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            suggest_neutral_points(np.full((200, 300, 3), 12000, dtype=np.uint16), Cancel())
