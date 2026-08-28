import tempfile
import unittest
from pathlib import Path

import numpy as np

from meteor_learning import _atomic_dataset


class CumulativeDatasetTests(unittest.TestCase):
    def test_atomic_dataset_roundtrip_keeps_training_arrays(self):
        dataset = {
            "x": np.arange(24, dtype=np.float32).reshape(6, 4),
            "y": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8),
            "groups": np.asarray(["a", "a", "b", "b", "c", "c"]),
            "legacy": np.linspace(0, 1, 6, dtype=np.float32),
            "metadata": np.asarray([f"item-{index}" for index in range(6)]),
        }
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "candidate_dataset_user.npz"
            _atomic_dataset(target, dataset)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_name(target.stem + ".writing.npz").exists())
            with np.load(target, allow_pickle=False) as loaded:
                for key, expected in dataset.items():
                    np.testing.assert_array_equal(loaded[key], expected)


if __name__ == "__main__":
    unittest.main()
