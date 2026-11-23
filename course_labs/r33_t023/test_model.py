# Unit tests for R33‑T023 fixture, split, scaler and nearest‑centroid
import unittest
from typing import List

from course_labs.r33_t023.data_fixture import generate_fixture
from course_labs.r33_t023.split_manifest import create_split
from course_labs.r33_t023.model import SimpleScaler, NearestCentroid

class TestFixture(unittest.TestCase):
    def test_fixture_content(self):
        records = generate_fixture()
        self.assertEqual(len(records), 6)
        expected = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (5.0, 5.0),
            (5.0, 6.0),
            (6.0, 5.0),
        ]
        for rec, (x1, x2) in zip(records, expected):
            self.assertEqual(rec["x1"], x1)
            self.assertEqual(rec["x2"], x2)
        ids = {r["record_id"] for r in records}
        self.assertEqual(len(ids), 6)

class TestSplit(unittest.TestCase):
    def test_deterministic_split(self):
        split1 = create_split(seed=123)
        split2 = create_split(seed=123)
        self.assertEqual(split1, split2)
        # 4 train, 1 val, 1 test
        self.assertEqual(len(split1["train"]), 4)
        self.assertEqual(len(split1["val"]), 1)
        self.assertEqual(len(split1["test"]), 1)
        # disjointness
        all_ids = set(split1["train"] + split1["val"] + split1["test"])
        self.assertEqual(len(all_ids), 6)

class TestModel(unittest.TestCase):
    def setUp(self):
        self.records = generate_fixture()
        self.split = create_split(seed=42)
        self.train = [r for r in self.records if r["record_id"] in self.split["train"]]
        self.val = [r for r in self.records if r["record_id"] in self.split["val"]]
        self.test = [r for r in self.records if r["record_id"] in self.split["test"]]

    def test_scaler_fit_transform(self):
        scaler = SimpleScaler()
        scaler.fit(self.train)
        # Transform on training data should succeed
        scaled = scaler.transform(self.train)
        self.assertEqual(len(scaled), len(self.train))
        # Transform on unseen record should raise
        with self.assertRaises(ValueError):
            scaler.transform(self.val)

    def test_nearest_centroid(self):
        nc = NearestCentroid()
        assignments = nc.assign(self.records)
        # Expected assignments based on fixed centroids
        expected = [0, 0, 0, 1, 1, 1]
        self.assertListEqual(assignments, expected)

    def test_failure_contamination(self):
        # Corrupt a training point to break scaler variance
        bad_train = self.train.copy()
        bad_train[0]["x1"] = 100.0
        scaler = SimpleScaler()
        scaler.fit(bad_train)
        # Ensure transform still works on the corrupted training set
        scaled = scaler.transform(bad_train)
        self.assertEqual(len(scaled), len(bad_train))

if __name__ == "__main__":
    unittest.main()
