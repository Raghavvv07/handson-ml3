# Synthetic fixture for R33-T024
"""Utility to generate the six 4×4 grayscale images used for the MLP classification task.

Each record is a dict with:
- ``record_id``: a unique identifier ("rec-0" … "rec-5")
- ``group_id`` : a group identifier – we simply reuse the label as the group.
- ``pixels``   : a flat list of 16 integers (0‑255) representing the image.
- ``label``    : binary class label (0 for all‑zero image, 1 for diagonal image).

The eight rows are:
```
rec-0 : all zeros, label 0
rec-1 : all zeros, label 0
rec-2 : all zeros, label 0
rec-3 : white diagonal, label 1
rec-4 : white diagonal, label 1
rec-5 : white diagonal, label 1
```
This data is synthetic and only for pipeline verification.
"""
from __future__ import annotations
from typing import List, Dict

def _zero_image() -> List[int]:
    return [0] * 16

def _diag_image() -> List[int]:
    img = []
    for r in range(4):
        for c in range(4):
            img.append(255 if r == c else 0)
    return img

def generate_fixture() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    # three zero‑label images
    for i in range(3):
        records.append({
            "record_id": f"rec-{len(records)}",
            "group_id": "group-0",
            "pixels": _zero_image(),
            "label": 0,
        })
    # three diagonal images
    for i in range(3):
        records.append({
            "record_id": f"rec-{len(records)}",
            "group_id": "group-1",
            "pixels": _diag_image(),
            "label": 1,
        })
    return records

if __name__ == "__main__":
    import json, sys
    json.dump(generate_fixture(), sys.stdout, indent=2)
