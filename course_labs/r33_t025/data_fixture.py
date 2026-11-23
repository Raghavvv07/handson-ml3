"""Synthetic fixture for R33‑T025 – nearest‑centroid unsupervised task.

Returns a list of six dicts, each with stable ``record_id``, ``x1`` and ``x2``.
The true cluster label is derived from the nearest of the two fixed centroids
(0,0) and (5,5).
"""

from __future__ import annotations
from typing import List, Dict

def generate_fixture() -> List[Dict[str, int]]:
    """Create the six points with their true clusters.

    The returned dicts contain ``record_id``, ``x1``, ``x2`` and the ground‑truth
    ``true_label`` (0 for centroid (0,0), 1 for centroid (5,5)).
    """
    points = [
        (0, 0),  # id 0 → centroid 0
        (0, 1),  # id 1 → centroid 0
        (1, 0),  # id 2 → centroid 0
        (5, 5),  # id 3 → centroid 1
        (5, 6),  # id 4 → centroid 1
        (6, 5),  # id 5 → centroid 1
    ]
    records: List[Dict[str, int]] = []
    for i, (x1, x2) in enumerate(points):
        true_label = 0 if i < 3 else 1
        records.append({"record_id": i, "x1": x1, "x2": x2, "true_label": true_label})
    return records

if __name__ == "__main__":
    for r in generate_fixture():
        print(r)
