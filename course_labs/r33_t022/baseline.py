# Baseline model for R33‑T022
"""Simple majority‑class baseline for the synthetic classification fixture.

The training step computes the most frequent label in the training set. In case of a tie
(the synthetic data has exactly four 0s and four 1s), the implementation chooses the
smaller label (0) for determinism.

The ``predict`` function returns the constant majority label for any input records.
"""
from __future__ import annotations
from typing import List, Dict

def train_majority(train_records: List[Dict[str, object]]) -> int:
    """Return the majority class label from ``train_records``.

    Parameters
    ----------
    train_records: list of dicts
        Each dict must contain a ``label`` key with an integer class label.

    Returns
    -------
    int
        The majority label (0 in case of a tie).
    """
    # Count occurrences of each label
    counts: dict[int, int] = {}
    for rec in train_records:
        lbl = int(rec["label"])  # ensure int
        counts[lbl] = counts.get(lbl, 0) + 1
    # Find max count; tie‑break by smallest label
    max_count = max(counts.values())
    candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
    return min(candidates)

def predict(constant_label: int, records: List[Dict[str, object]]) -> List[int]:
    """Predict the constant ``constant_label`` for each record.

    Returns a list of predicted labels matching the order of ``records``.
    """
    return [constant_label for _ in records]

def accuracy(true_labels: List[int], pred_labels: List[int]) -> float:
    """Simple accuracy metric.

    Returns the proportion of correct predictions.
    """
    if not true_labels:
        return 0.0
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels)

if __name__ == "__main__":
    from .data_fixture import generate_fixture
    from .split_manifest import create_split
    # Example run
    data = generate_fixture()
    split = create_split()
    train = [r for r in data if r["record_id"] in split["train"]]
    val = [r for r in data if r["record_id"] in split["val"]]
    maj = train_majority(train)
    preds = predict(maj, val)
    true = [r["label"] for r in val]
    print("majority label", maj, "accuracy", accuracy(true, preds))
