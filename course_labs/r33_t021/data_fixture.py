# Synthetic fixture for R33-T021
"""Utility to generate a tiny synthetic image dataset.

Each record is a dictionary with the following fields:
- ``record_id``: a unique string identifier ("rec-0" … "rec-5")
- ``group_id`` : the group the record belongs to (same as its label for simplicity)
- ``pixels``   : a list of 16 integers (0‑255) representing a 4×4 grayscale image
- ``label``    : 0 for all‑zero images, 1 for images with a white diagonal

The function returns a list of six records.
"""
from __future__ import annotations
from typing import List, Dict

def _zero_image() -> List[int]:
    """Return a flat list of 16 zeros (4×4 image)."""
    return [0] * 16

def _diag_image() -> List[int]:
    """Return a 4×4 image where the main diagonal is 255 and the rest are 0.

    The flat list is row‑major order.
    """
    img = []
    for r in range(4):
        for c in range(4):
            img.append(255 if r == c else 0)
    return img

def generate_fixture() -> List[Dict[str, object]]:
    """Create the six synthetic records.

    Returns
    -------
    List[Dict]
        List of six dicts, each containing ``record_id``, ``group_id``,
        ``pixels`` and ``label``.
    """
    records: List[Dict[str, object]] = []
    # three zero‑label images (label 0, group 0)
    for i in range(3):
        rec = {
            "record_id": f"rec-{len(records)}",
            "group_id": "group-0",
            "pixels": _zero_image(),
            "label": 0,
        }
        records.append(rec)
    # three diagonal images (label 1, group 1)
    for i in range(3):
        rec = {
            "record_id": f"rec-{len(records)}",
            "group_id": "group-1",
            "pixels": _diag_image(),
            "label": 1,
        }
        records.append(rec)
    return records

if __name__ == "__main__":
    import json, sys
    json.dump(generate_fixture(), sys.stdout, indent=2)
