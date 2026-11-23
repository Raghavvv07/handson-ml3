# Simple scaling and nearest‑centroid model for R33‑T023
"""Provides a minimal scaler and a nearest‑centroid assignment.

* ``SimpleScaler`` computes per‑feature mean and standard deviation on the
  training records and stores them. ``transform`` can only be called on the
  exact same records that were used for fitting – any other ``record_id``
  results in a ``ValueError`` (this satisfies the requirement that held‑out
  data cannot be transformed through the public interface).

* ``NearestCentroid`` uses the two fixed centroids (0, 0) and (5, 5) and
  returns the index of the nearest centroid for each input point.

Both classes operate on the synthetic fixture defined in
``data_fixture.py``. The implementation is deliberately simple and CPU‑only.
"""
from __future__ import annotations
import math
from typing import List, Dict, Tuple

class SimpleScaler:
    """Fit mean and standard deviation on training records.

    The scaler expects records with numeric ``x1`` and ``x2`` fields.
    After ``fit`` the attributes ``mean`` and ``std`` are lists
    ``[mean_x1, mean_x2]`` and ``[std_x1, std_x2]``.
    ``transform`` returns a list of ``(scaled_x1, scaled_x2)`` tuples for the
    provided records, but only if *all* ``record_id`` values are present in the
    training set used for ``fit``.
    """
    def __init__(self) -> None:
        self.mean: List[float] = []
        self.std: List[float] = []
        self._train_ids: set[str] = set()

    def fit(self, records: List[Dict[str, object]]) -> None:
        xs = [float(r["x1"]) for r in records]
        ys = [float(r["x2"]) for r in records]
        n = len(records)
        if n == 0:
            raise ValueError("Cannot fit on empty data")
        self.mean = [sum(xs) / n, sum(ys) / n]
        # unbiased standard deviation (divide by n, not n‑1 – sufficient for this task)
        self.std = [math.sqrt(sum((x - self.mean[0]) ** 2 for x in xs) / n) or 1.0,
                    math.sqrt(sum((y - self.mean[1]) ** 2 for y in ys) / n) or 1.0]
        self._train_ids = {r["record_id"] for r in records}

    def transform(self, records: List[Dict[str, object]]) -> List[Tuple[float, float]]:
        # Ensure only training IDs are present
        ids = {r["record_id"] for r in records}
        if not ids.issubset(self._train_ids):
            raise ValueError("Transform called on records not seen during fit")
        result: List[Tuple[float, float]] = []
        for r in records:
            x = (float(r["x1"]) - self.mean[0]) / self.std[0]
            y = (float(r["x2"]) - self.mean[1]) / self.std[1]
            result.append((x, y))
        return result

class NearestCentroid:
    """Assign each point to the nearest of two fixed centroids.

    Centroids are (0, 0) and (5, 5). The method ``assign`` returns a list of
    integer labels ``0`` or ``1`` corresponding to the nearest centroid for each
    input record.
    """
    def __init__(self) -> None:
        self.centroids = [(0.0, 0.0), (5.0, 5.0)]

    @staticmethod
    def _euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def assign(self, records: List[Dict[str, object]]) -> List[int]:
        assignments: List[int] = []
        for r in records:
            point = (float(r["x1"]), float(r["x2"]))
            d0 = self._euclidean(point, self.centroids[0])
            d1 = self._euclidean(point, self.centroids[1])
            assignments.append(0 if d0 <= d1 else 1)
        return assignments

if __name__ == "__main__":
    from .data_fixture import generate_fixture
    data = generate_fixture()
    scaler = SimpleScaler()
    scaler.fit(data)
    print("means", scaler.mean, "stds", scaler.std)
    print("scaled", scaler.transform(data))
    nc = NearestCentroid()
    print("centroid assignments", nc.assign(data))
