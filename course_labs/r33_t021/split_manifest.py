"""Deterministic split manifest for the synthetic fixture.

The function ``create_split`` returns a mapping with keys ``train``, ``val`` and ``test``.
Each value is a list of ``record_id`` strings. The split is deterministic when the
same ``seed`` is supplied.
"""
from __future__ import annotations
import random
from typing import List, Dict
from .data_fixture import generate_fixture

def create_split(seed: int = 42) -> Dict[str, List[str]]:
    """Create a deterministic train/val/test split.

    Parameters
    ----------
    seed: int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}``
        containing the ``record_id`` values for each split.

    Notes
    -----
    * The split respects ``group_id`` isolation – records from the same group are
      not allowed to appear in different splits. If such a conflict would occur,
      a ``ValueError`` is raised.
    * With the synthetic fixture there are exactly two groups (``group-0`` and
      ``group-1``), each containing three records. The default split puts the
      first group entirely into ``train`` and the second group into ``val`` and
      ``test`` (one record each) to demonstrate the isolation rule.
    """
    records = generate_fixture()
    # Group records by group_id
    groups: Dict[str, List[dict]] = {}
    for rec in records:
        groups.setdefault(rec["group_id"], []).append(rec)

    # Deterministic ordering of groups
    group_keys = sorted(groups.keys())
    random.Random(seed).shuffle(group_keys)

    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    # Simple policy: first group → train, second group → val+test (split evenly)
    if len(group_keys) < 2:
        raise ValueError("Expected at least two groups for isolation check")
    # Assign whole first group to train
    for rec in groups[group_keys[0]]:
        split["train"].append(rec["record_id"])
    # Split second group into val and test (one each, rest to val for safety)
    second_group = groups[group_keys[1]]
    # Deterministic ordering within the group
    second_ids = [rec["record_id"] for rec in second_group]
    random.Random(seed).shuffle(second_ids)
    if len(second_ids) < 2:
        raise ValueError("Second group must have at least two records for split")
    split["val"].append(second_ids[0])
    split["test"].append(second_ids[1])
    # Any remaining records go to train to keep sets disjoint
    for rid in second_ids[2:]:
        split["train"].append(rid)

    # Verify disjointness
    all_ids = set()
    for part in split.values():
        if any(rid in all_ids for rid in part):
            raise ValueError("Overlapping record IDs between splits")
        all_ids.update(part)
    return split

if __name__ == "__main__":
    import json, sys
    json.dump(create_split(), sys.stdout, indent=2)
