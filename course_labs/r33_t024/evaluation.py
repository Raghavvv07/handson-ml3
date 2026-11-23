# Evaluation utilities for R33‑T024
"""Simple classification metrics used to evaluate the synthetic image fixture.

The functions are deliberately lightweight and work on plain Python lists or
NumPy‑compatible sequences. They explicitly handle the zero‑denominator case by
returning ``0.0`` when the denominator would be zero.

* ``precision`` – TP / (TP + FP)
* ``recall``    – TP / (TP + FN)

Both return ``0.0`` when the denominator is zero (e.g., no positive predictions
for precision or no positive true labels for recall).
"""
from __future__ import annotations
from typing import Sequence

def _counts(truth: Sequence[int], pred: Sequence[int]) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) counts for binary labels.

    ``truth`` and ``pred`` must contain only 0 or 1.
    """
    if len(truth) != len(pred):
        raise ValueError("truth and pred must have the same length")
    tp = fp = fn = tn = 0
    for t, p in zip(truth, pred):
        if p == 1:
            if t == 1:
                tp += 1
            else:
                fp += 1
        else:  # p == 0
            if t == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def precision(truth: Sequence[int], pred: Sequence[int]) -> float:
    """Compute precision = TP / (TP + FP). Returns 0.0 if denominator is 0.
    """
    tp, fp, _, _ = _counts(truth, pred)
    denom = tp + fp
    return tp / denom if denom != 0 else 0.0

def recall(truth: Sequence[int], pred: Sequence[int]) -> float:
    """Compute recall = TP / (TP + FN). Returns 0.0 if denominator is 0.
    """
    tp, _, fn, _ = _counts(truth, pred)
    denom = tp + fn
    return tp / denom if denom != 0 else 0.0

if __name__ == "__main__":
    example_truth = [1, 1, 0, 0]
    example_pred = [1, 0, 1, 0]
    print("precision", precision(example_truth, example_pred))
    print("recall", recall(example_truth, example_pred))
