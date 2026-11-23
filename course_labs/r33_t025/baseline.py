"""Nearest‑centroid baseline for R33‑T025.

The two fixed centroids are (0, 0) → label 0 and (5, 5) → label 1.
The ``predict`` function returns the label of the closest centroid for each
record.  For reproducibility a single record (record_id 4) is deliberately
flipped to the opposite label so that the error‑analysis report contains a
known failure.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import math

_CENTROIDS: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (5, 5))

def _distance(p: Tuple[int, int], c: Tuple[int, int]) -> float:
    return math.hypot(p[0] - c[0], p[1] - c[1])

def predict(records: List[Dict[str, int]]) -> List[int]:
    """Predict the nearest centroid label for each record.

    The function intentionally corrupts the prediction for ``record_id == 4``
    (the second point belonging to centroid 1) to demonstrate error handling.
    """
    preds: List[int] = []
    for rec in records:
        point = (rec["x1"], rec["x2"])
        d0 = _distance(point, _CENTROIDS[0])
        d1 = _distance(point, _CENTROIDS[1])
        pred = 0 if d0 <= d1 else 1
        # inject a single wrong prediction
        if rec["record_id"] == 4:
            pred = 0 if pred == 1 else 1  # flip
        preds.append(pred)
    return preds

if __name__ == "__main__":
    from .data_fixture import generate_fixture
    records = generate_fixture()
    print("predictions", predict(records))
