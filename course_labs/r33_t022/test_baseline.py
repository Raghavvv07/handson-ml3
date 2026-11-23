# Unit tests for R33‑T022 synthetic fixture, split, and baseline
import unittest
from typing import List

from course_labs.r33_t022.data_fixture import generate_fixture
from course_labs.r33_t022.split_manifest import create_split
from course_labs.r33_t022.baseline import train_majority, predict, accuracy

class TestFixture(unittest.TestCase):
    def test_fixture_content(self):
        records = generate_fixture()
        self.assertEqual(len(records), 8)
        # Check x1 values and labels
        expected = [(i, 1, 0 if i < 4 else 1) for i in range(8)]
        for rec, (exp_x1, exp_x2, exp_label) in zip(records, expected):
            self.assertEqual(rec["x1"], exp_x1)
            self.assertEqual(rec["x2"], exp_x2)
            self.assertEqual(rec["label"], exp_label)
        # Record IDs are unique
        ids = [r["record_id"] for r in records]
        self.assertEqual(len(set(ids)), 8)

class TestSplit(unittest.TestCase):
    def test_stratified_split(self):
        split = create_split(seed=123)
        # Should contain 4 train, 2 val, 2 test
        self.assertEqual(len(split["train"]), 4)
        self.assertEqual(len(split["val"]), 2)
        self.assertEqual(len(split["test"]), 2)
        # Disjointness
        all_ids = set(split["train"] + split["val"] + split["test"])
        self.assertEqual(len(all_ids), 8)
        # Stratification: each split has at least one of each label where possible
        records = {r["record_id"]: r for r in generate_fixture()}
        for part in ("val", "test"):
            labels = {records[rid]["label"] for rid in split[part]}
            # With our allocation each part gets one 0 and one 1
            self.assertEqual(labels, {0, 1})

    def test_determinism(self):
        s1 = create_split(seed=42)
        s2 = create_split(seed=42)
        self.assertEqual(s1, s2)

class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.records = generate_fixture()
        self.split = create_split(seed=42)
        self.train = [r for r in self.records if r["record_id"] in self.split["train"]]
        self.val = [r for r in self.records if r["record_id"] in self.split["val"]]
        self.test = [r for r in self.records if r["record_id"] in self.split["test"]]

    def test_majority_label(self):
        maj = train_majority(self.train)
        # Training set contains two 0s and two 1s -> tie -> smaller label 0
        self.assertEqual(maj, 0)

    def test_accuracy(self):
        maj = train_majority(self.train)
        preds_val = predict(maj, self.val)
        true_val = [r["label"] for r in self.val]
        acc_val = accuracy(true_val, preds_val)
        # With majority=0 and val contains one 0 and one 1, accuracy = 0.5
        self.assertAlmostEqual(acc_val, 0.5)
        # Same for test split
        preds_test = predict(maj, self.test)
        true_test = [r["label"] for r in self.test]
        acc_test = accuracy(true_test, preds_test)
        self.assertAlmostEqual(acc_test, 0.5)

    def test_failure_case(self):
        # Corrupt training data to force a tie‑break to label 1
        bad_train = self.train.copy()
        # Flip a label to make three 1s and one 0
        bad_train[0]["label"] = 1
        maj = train_majority(bad_train)
        self.assertEqual(maj, 1)

if __name__ == "__main__":
    unittest.main()
