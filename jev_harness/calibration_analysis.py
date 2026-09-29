"""ECE and isotonic recalibration for the hallucination-judge post.

Builds on jev_harness.stats.calibration_buckets() rather than duplicating the
bucketing logic. Every function here expects rows shaped like stats.py's
convention: each row has 'confidence' (probability assigned to the predicted
class, 0.5-1.0) and 'correct' (bool: did the predicted class match the true
label). Turning a raw Jev `noul` or LLM-judge verdict into that shape is the
caller's job (see hallucination/analyze.py) - this module is judge-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from sklearn.isotonic import IsotonicRegression

from .stats import calibration_buckets


def expected_calibration_error(rows: list[dict[str, Any]], bin_width: float = 0.1) -> float:
    """Weighted mean |confidence - accuracy| across bins, weighted by bin population."""
    buckets = calibration_buckets(rows, bin_width=bin_width)
    total_n = sum(b["n"] for b in buckets)
    if total_n == 0:
        return float("nan")
    return sum(b["n"] * abs(b["mean_confidence"] - b["empirical_accuracy"]) for b in buckets) / total_n


def split_train_test(
    rows: list[dict[str, Any]], n_train: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


def fit_isotonic(rows: list[dict[str, Any]]) -> IsotonicRegression:
    x = [r["confidence"] for r in rows]
    y = [1.0 if r["correct"] else 0.0 for r in rows]
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(x, y)
    return model


def apply_calibrator(
    calibrator: IsotonicRegression, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Returns new rows with 'confidence' replaced by the recalibrated value."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["confidence"] = float(calibrator.predict([r["confidence"]])[0])
        out.append(r2)
    return out


@dataclass
class RecalibrationResult:
    n_train: int
    ece_before: float
    ece_after: float
    calibrator: IsotonicRegression


def recalibrate(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    n_train_subset: int | None = None,
) -> RecalibrationResult:
    train_subset = train_rows[:n_train_subset] if n_train_subset else train_rows
    ece_before = expected_calibration_error(test_rows)
    calibrator = fit_isotonic(train_subset)
    recalibrated_test = apply_calibrator(calibrator, test_rows)
    ece_after = expected_calibration_error(recalibrated_test)
    return RecalibrationResult(
        n_train=len(train_subset),
        ece_before=ece_before,
        ece_after=ece_after,
        calibrator=calibrator,
    )


def learning_curve(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    sizes: tuple[int, ...] = (50, 100, 200, 400),
) -> list[dict[str, Any]]:
    out = []
    for n in sizes:
        if n > len(train_rows):
            continue
        result = recalibrate(train_rows, test_rows, n_train_subset=n)
        out.append(
            {"n_train": result.n_train, "ece_before": result.ece_before, "ece_after": result.ece_after}
        )
    return out
