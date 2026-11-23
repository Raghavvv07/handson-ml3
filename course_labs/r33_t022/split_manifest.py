# Deterministic stratified split manifest for R33‑T022 synthetic fixture
"""Create a reproducible train/validation/test split.

The split is *stratified* by the ``label`` field so that each split
contains a proportionate number of 0s and 1s.  With eight rows we allocate
4 records to training, 2 to validation and 2 to test.

The function returns a dict with keys ``train``, ``val`` and ``test``
containing the ``record_id`` strings for the respective splits.
"""
from __future__ import annotations
import random
from typing import List, Dict
from .data_fixture import generate_fixture

def create_split(seed: int = 42) -> Dict[str, List[str]]:
    """Return a deterministic stratified split.

    Parameters
    ----------
    seed: int, default 42
        Random seed used to shuffle the records within each label.

    Returns
    -------
    dict
        Mapping ``{"train": [...], "val": [...], "test": [...]}`` of record IDs.
    """
    records = generate_fixture()
    # Group by label
    groups: Dict[int, List[Dict[str, object]]] = {}
    for rec in records:
        groups.setdefault(rec["label"], []).append(rec)
    # Shuffle deterministically inside each label group
    rnd = random.Random(seed)
    for lst in groups.values():
        rnd.shuffle(lst)
    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    # Allocation plan: 2 records per label to train, 1 to val, 1 to test
    for label, recs in groups.items():
        if len(recs) != 4:
            raise ValueError("Expected exactly four records per label for stratified split.")
        # first two -> train, third -> val, fourth -> test
        split["train"].extend([r["record_id"] for r in recs[:2]])
        split["val"].append(recs[2]["record_id"])
        split["test"].append(recs[3]["record_id"])
    # Verify disjointness
    all_ids = set()
    for part in split.values():
        if any(r in all_ids for r in part):
            raise ValueError("Overlapping record IDs between splits")
        all_ids.update(part)
    return split

if __name__ == "__main__":
    import json, sys
    json.dump(create_split(), sys.stdout, indent=2)
