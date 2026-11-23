"""Error‑analysis report for the R33‑T025 nearest‑centroid task.

The function ``generate_report`` takes the fixture records and the list of
predictions, compares them to the ground‑truth ``true_label`` and returns a
list of dictionaries with the columns required by the assignment:

* ``record_id`` – stable identifier
* ``expected``   – true label
* ``predicted``  – model prediction (with the intentional error)
* ``reason``     – ``"correct"`` or ``"injection_error"``

The report is deliberately small; downstream unit tests assert that the
injected wrong prediction (record_id 4) is present with the correct reason.
"""

from __future__ import annotations
from typing import List, Dict

def generate_report(records: List[Dict[str, int]], predictions: List[int]) -> List[Dict[str, object]]:
    """Create an error‑analysis report.

    Parameters
    ----------
    records: list of dicts containing ``record_id`` and ``true_label``.
    predictions: list of int predictions, same order as ``records``.

    Returns
    -------
    list of dicts with keys ``record_id``, ``expected``, ``predicted`` and
    ``reason``.
    """
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")
    report: List[Dict[str, object]] = []
    for rec, pred in zip(records, predictions):
        expected = rec["true_label"]
        reason = "correct" if pred == expected else "injection_error"
        report.append({
            "record_id": rec["record_id"],
            "expected": expected,
            "predicted": pred,
            "reason": reason,
        })
    return report

if __name__ == "__main__":
    from .data_fixture import generate_fixture
    from .baseline import predict
    records = generate_fixture()
    preds = predict(records)
    for row in generate_report(records, preds):
        print(row)
