"""Grading and aggregate stats over flattened result rows.

A "row" is one answered question, as produced by runner.run_batch: a dict
with at least type, value, confidence, expected, correct.
"""

from __future__ import annotations

from typing import Any


def grade(answer_type: str, value: Any, expected: Any) -> bool | None:
    """Return True/False if `expected` states a known-correct answer, else None."""
    if expected is None or value is None:
        return None
    if answer_type == "choice":
        return value == expected
    if answer_type == "noul":
        return (value >= 0.5) == bool(expected)
    if answer_type == "score":
        return round(value) == int(expected)
    raise ValueError(f"unknown answer type: {answer_type!r}")


def confidence_histogram(confidences: list[float], bin_width: float = 0.1) -> dict[str, int]:
    bins: dict[str, int] = {}
    for c in confidences:
        lo = min(int(c / bin_width) * bin_width, 1 - bin_width)
        key = f"{lo:.1f}-{lo + bin_width:.1f}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items()))


def calibration_buckets(rows: list[dict[str, Any]], bin_width: float = 0.1) -> list[dict[str, Any]]:
    """Bucket graded rows by reported confidence and compare to empirical accuracy.

    This is the core calibration check: for a well-calibrated model, rows
    with confidence in [0.7, 0.8) should be correct about 70-80% of the time.
    """
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("confidence") is None or row.get("correct") is None:
            continue
        lo = min(int(row["confidence"] / bin_width) * bin_width, 1 - bin_width)
        buckets.setdefault(lo, []).append(row)

    out = []
    for lo in sorted(buckets):
        bucket_rows = buckets[lo]
        n = len(bucket_rows)
        empirical_accuracy = sum(1 for r in bucket_rows if r["correct"]) / n
        mean_confidence = sum(r["confidence"] for r in bucket_rows) / n
        out.append(
            {
                "bucket": f"{lo:.1f}-{lo + bin_width:.1f}",
                "n": n,
                "mean_confidence": round(mean_confidence, 4),
                "empirical_accuracy": round(empirical_accuracy, 4),
                "gap": round(mean_confidence - empirical_accuracy, 4),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if r.get("correct") is not None]
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    errors = [r for r in rows if r.get("error")]

    return {
        "n_rows": len(rows),
        "n_graded": len(graded),
        "n_errors": len(errors),
        "accuracy": (round(sum(1 for r in graded if r["correct"]) / len(graded), 4) if graded else None),
        "confidence_histogram": confidence_histogram(confidences) if confidences else {},
        "calibration_buckets": calibration_buckets(rows),
    }
