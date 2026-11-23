# Unit tests for R33‑T024 synthetic fixture and evaluation utilities
import unittest
from typing import List

from course_labs.r33_t024.data_fixture import generate_fixture
from course_labs.r33_t024.split_manifest import create_split
from course_labs.r33_t024.evaluation import precision, recall

class TestFixture(unittest.TestCase):
    def test_fixture_content(self):
        records = generate_fixture()
        self.assertEqual(len(records), 6)
        # Verify three zero images and three diagonal images
        zero_pixels = [0] * 16
        diag_pixels = []
        for r in range(4):
            for c in range(4):
                diag_pixels.append(255 if r == c else 0)
        zeros = [r for r in records if r["label"] == 0]
        diags = [r for r in records if r["label"] == 1]
        self.assertEqual(len(zeros), 3)
        self.assertEqual(len(diags), 3)
        for rec in zeros:
            self.assertEqual(rec["pixels"], zero_pixels)
        for rec in diags:
            self.assertEqual(rec["pixels"], diag_pixels)
        # Record IDs unique
        ids = [r["record_id"] for r in records]
        self.assertEqual(len(set(ids)), 6)

class TestSplit(unittest.TestCase):
    def test_deterministic(self):
        split1 = create_split(seed=123)
        split2 = create_split(seed=123)
        self.assertEqual(split1, split2)
        self.assertEqual(len(split1["train"]), 4)
        self.assertEqual(len(split1["val"]), 1)
        self.assertEqual(len(split1["test"]), 1)
        # Ensure disjointness
        all_ids = set(split1["train"] + split1["val"] + split1["test"])
        self.assertEqual(len(all_ids), 6)

class TestEvaluation(unittest.TestCase):
    def test_precision_recall_example(self):
        truth = [1, 1, 0, 0]
        pred =  [1, 0, 1, 0]
        self.assertAlmostEqual(precision(truth, pred), 0.5)
        self.assertAlmostEqual(recall(truth, pred), 0.5)

    def test_zero_denominator(self):
        # No positive predictions -> precision denominator zero
        truth = [0, 0, 0, 0]
        pred =  [0, 0, 0, 0]
        self.assertEqual(precision(truth, pred), 0.0)
        # No positive truth -> recall denominator zero
        self.assertEqual(recall(truth, pred), 0.0)

    def test_mismatched_length(self):
        with self.assertRaises(ValueError):
            precision([1, 0], [1])
        with self.assertRaises(ValueError):
            recall([1, 0], [1])

if __name__ == "__main__":
    unittest.main()
