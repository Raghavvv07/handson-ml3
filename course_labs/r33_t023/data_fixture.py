# Synthetic fixture for R33-T023
"""Utility to generate a tiny 2‑D synthetic dataset for a dimensionality‑reduction task.

Each record is a dictionary with the following keys:
- ``record_id``: a stable unique identifier ("rec-0" … "rec-5")
- ``x1``       : first coordinate (float)
- ``x2``       : second coordinate (float)

The six points are:
```
rec-0 : (0, 0)
rec-1 : (0, 1)
rec-2 : (1, 0)
rec-3 : (5, 5)
rec-4 : (5, 6)
rec-5 : (6, 5)
```
These are synthetic and intended solely for pipeline verification.
"""
from __future__ import annotations
from typing import List, Dict

def generate_fixture() -> List[Dict[str, object]]:
    """Return the six fixed points with stable IDs."""
    points = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (5.0, 5.0),
        (5.0, 6.0),
        (6.0, 5.0),
    ]
    records = []
    for i, (x1, x2) in enumerate(points):
        rec = {
            "record_id": f"rec-{i}",
            "x1": x1,
            "x2": x2,
        }
        records.append(rec)
    return records

if __name__ == "__main__":
    import json, sys
    json.dump(generate_fixture(), sys.stdout, indent=2)
