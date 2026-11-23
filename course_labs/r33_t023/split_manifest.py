# Deterministic split manifest for R33-T023
"""Create a reproducible train/validation/test split for the synthetic 2‑D points.

The function returns a dict with keys ``train``, ``val`` and ``test`` containing the
``record_id`` strings. The split is deterministic given the ``seed``.
"""
from __future__ import annotations
import random
from typing import List, Dict
from .data_fixture import generate_fixture

def create_split(seed: int = 42) -> Dict[str, List[str]]:
    """Return a deterministic split.

    Parameters
    ----------
    seed: int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of record IDs.
    """
    records = generate_fixture()
    ids = [r["record_id"] for r in records]
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    # Simple allocation: 4 train, 1 val, 1 test
    split = {
        "train": ids[:4],
        "val": ids[4:5],
        "test": ids[5:6],
    }
    # Verify disjointness
    all_ids = set()
    for part in split.values():
        if any(i in all_ids for i in part):
            raise ValueError("Overlapping record IDs between splits")
        all_ids.update(part)
    return split

if __name__ == "__main__":
    import json, sys
    json.dump(create_split(), sys.stdout, indent=2)
