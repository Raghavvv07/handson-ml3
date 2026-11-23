# Deterministic split manifest for R33‑T024
"""Create a reproducible train/validation/test split for the synthetic image fixture.

We allocate 4 records to training, 1 to validation and 1 to test, shuffling with a fixed seed.
"""
from __future__ import annotations
import random
from typing import List, Dict
from .data_fixture import generate_fixture

def create_split(seed: int = 42) -> Dict[str, List[str]]:
    records = generate_fixture()
    ids = [r["record_id"] for r in records]
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    return {
        "train": ids[:4],
        "val": ids[4:5],
        "test": ids[5:6],
    }

if __name__ == "__main__":
    import json, sys
    json.dump(create_split(), sys.stdout, indent=2)
