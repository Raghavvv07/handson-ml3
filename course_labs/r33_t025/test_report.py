import unittest
from typing import List, Dict

from .data_fixture import generate_fixture
from .split_manifest import stratified_split
from .baseline import predict
from .report import generate_report

class TestFixture(unittest.TestCase):
    def test_fixture_content(self):
        records = generate_fixture()
        self.assertEqual(len(records), 6)
        # expected true labels based on position
        expected_labels = [0, 0, 0, 1, 1, 1]
        for rec, exp in zip(records, expected_labels):
            self.assertEqual(rec["true_label"], exp)
        # IDs are stable and unique
        ids = [r["record_id"] for r in records]
        self.assertEqual(len(set(ids)), 6)

class TestSplit(unittest.TestCase):
    def test_stratified_split(self):
        records = generate_fixture()
        train, val, test = stratified_split(records)
        # check sizes
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 1)
        self.assertEqual(len(test), 1)
        # ensure disjoint IDs
        ids = {r["record_id"] for r in train + val + test}
        self.assertEqual(len(ids), 6)
        # each split should keep label distribution roughly balanced
        train_labels = {r["true_label"] for r in train}
        self.assertEqual(train_labels, {0, 1})

class TestBaselineAndReport(unittest.TestCase):
    def setUp(self):
        self.records = generate_fixture()
        self.preds = predict(self.records)
        self.report = generate_report(self.records, self.preds)

    def test_predictions_length(self):
        self.assertEqual(len(self.preds), len(self.records))

    def test_injected_error_present(self):
        # record_id 4 should be the only wrong prediction
        wrong = [row for row in self.report if row["reason"] == "injection_error"]
        self.assertEqual(len(wrong), 1)
        self.assertEqual(wrong[0]["record_id"], 4)
        # ensure the predicted label is opposite of expected
        self.assertNotEqual(wrong[0]["predicted"], wrong[0]["expected"])

    def test_all_correct_reason(self):
        correct = [row for row in self.report if row["reason"] == "correct"]
        self.assertEqual(len(correct), 5)
        # every correct row should have matching prediction
        for row in correct:
            self.assertEqual(row["predicted"], row["expected"])

    def test_mismatch_error(self):
        with self.assertRaises(ValueError):
            generate_report(self.records, self.preds[:-1])

if __name__ == "__main__":
    unittest.main()
