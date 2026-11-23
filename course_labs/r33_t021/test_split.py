# Unit tests for R33-T021 synthetic fixture and split manifest
import unittest
import json
from pathlib import Path

from course_labs.r33_t021.data_fixture import generate_fixture
from course_labs.r33_t021.split_manifest import create_split

class TestSyntheticFixture(unittest.TestCase):
    def test_fixture_length_and_labels(self):
        records = generate_fixture()
        self.assertEqual(len(records), 6, "Fixture should contain six records")
        labels = [r["label"] for r in records]
        self.assertCountEqual(labels, [0, 0, 0, 1, 1, 1])
        # Verify pixel patterns
        zero_img = [0] * 16
        diag_img = []
        for r in range(4):
            for c in range(4):
                diag_img.append(255 if r == c else 0)
        for rec in records:
            if rec["label"] == 0:
                self.assertEqual(rec["pixels"], zero_img)
            else:
                self.assertEqual(rec["pixels"], diag_img)

class TestSplitManifest(unittest.TestCase):
    def test_deterministic_split(self):
        split1 = create_split(seed=123)
        split2 = create_split(seed=123)
        self.assertEqual(split1, split2, "Splits must be deterministic for same seed")
        # Ensure disjointness
        all_ids = set(split1["train"] + split1["val"] + split1["test"])
        self.assertEqual(len(all_ids), len(split1["train"]) + len(split1["val"]) + len(split1["test"]))
        # Ensure at least one record in each split
        self.assertTrue(split1["train"], "Train split should not be empty")
        self.assertTrue(split1["val"], "Validation split should not be empty")
        self.assertTrue(split1["test"], "Test split should not be empty")

    def test_overlap_error(self):
        # Monkey‑patch generate_fixture to create overlapping groups
        original = generate_fixture
        def overlapping_fixture():
            # All records in same group – will cause overlap when we force split
            recs = original()
            for r in recs:
                r["group_id"] = "group-0"
            return recs
        # Replace function temporarily
        try:
            import importlib
            import course_labs.r33_t021.split_manifest as sm
            sm.generate_fixture = overlapping_fixture
            with self.assertRaises(ValueError):
                sm.create_split(seed=42)
        finally:
            sm.generate_fixture = original

if __name__ == "__main__":
    unittest.main()
