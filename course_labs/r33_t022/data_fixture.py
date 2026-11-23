# Synthetic fixture for R33-T022
"""Utility to generate a tiny synthetic dataset for a binary classification task.

Each record is a dict with the following fields:
- ``record_id``: a unique identifier string (e.g. "rec-0" … "rec-7")
- ``group_id`` : a group identifier – for this synthetic data we simply use the label as the group.
- ``x1``       : integer feature ranging from 0 to 7.
- ``x2``       : constant feature equal to 1.
- ``label``    : binary class label (0 or 1).

The fixture contains eight rows:
```
row   x1  x2  label
0     0   1   0
1     1   1   0
2     2   1   0
3     3   1   0
4     4   1   1
5     5   1   1
6     6   1   1
7     7   1   1
```
The data is **synthetic** and intended only for pipeline verification.
"""
from __future__ import annotations
from typing import List, Dict

def generate_fixture() -> List[Dict[str, object]]:
    """Return the eight synthetic records.

    The ``group_id`` mirrors the ``label`` value so that rows with the same
    label belong to the same group – this makes stratified splitting straightforward.
    """
    records: List[Dict[str, object]] = []
    x2 = 1
    # labels: first four are 0, next four are 1
    for i in range(8):
        label = 0 if i < 4 else 1
        rec = {
            "record_id": f"rec-{i}",
            "group_id": f"group-{label}",
            "x1": i,
            "x2": x2,
            "label": label,
        }
        records.append(rec)
    return records

if __name__ == "__main__":
    import json, sys
    json.dump(generate_fixture(), sys.stdout, indent=2)
