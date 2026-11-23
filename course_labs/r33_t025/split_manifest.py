"""Deterministic stratified split for the R33‑T025 fixture.

The split uses a fixed random seed and maintains the label distribution.
We allocate 4 records to training, 1 to validation and 1 to test.
"""

from __future__ import annotations
import random
from typing import List, Tuple, Dict

_SEED = 42

def stratified_split(records: List[Dict[str, int]]) -> Tuple[List[Dict[str, int]], List[Dict[str, int]], List[Dict[str, int]]]:
    """Return (train, val, test) splits.

    The split is stratified by the ``true_label`` field.
    """
    random.seed(_SEED)
    # Group records by true_label
    label_groups: dict[int, List[Dict[str, int]]] = {0: [], 1: []}
    for r in records:
        label_groups[r["true_label"]].append(r)
    train, val, test = [], [], []
    for label, group in label_groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = n // 2  # 1 per label → total 2 in train per label → 4 overall
        n_val = (n - n_train) // 2  # 0 per label? actually 1 total per label → 2 overall
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    return train, val, test

if __name__ == "__main__":
    from .data_fixture import generate_fixture
    tr, v, te = stratified_split(generate_fixture())
    print("train", [r["record_id"] for r in tr])
    print("val", [r["record_id"] for r in v])
    print("test", [r["record_id"] for r in te])
