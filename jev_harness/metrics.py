"""Ranking and separation metrics with seeded bootstrap intervals."""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Sequence


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    gains = [1.0 if d in relevant else 0.0 for d in ranked[:k]]
    ideal = _dcg([1.0] * min(len(relevant), k))
    return _dcg(gains) / ideal if ideal else 0.0


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg), ties count 0.5 (Mann-Whitney U / (n*m))."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.mean,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, 2.5th, 97.5th percentile) resampling items with replacement."""
    vals = list(values)
    rng = random.Random(seed)
    stats = sorted(stat([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n_boot))
    return stat(vals), stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) per item with a bootstrap interval; items are paired."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], n_boot=n_boot, seed=seed)


def auc_by_question(
    pos_by_q: Sequence[Sequence[float]], neg_by_q: Sequence[Sequence[float]]
) -> float:
    return auc([p for ps in pos_by_q for p in ps], [n for ns in neg_by_q for n in ns])


def auc_bootstrap(
    pos_by_q: Sequence[Sequence[float]],
    neg_by_q: Sequence[Sequence[float]],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUC with a bootstrap interval that resamples whole questions (pos and neg stay paired)."""
    n = len(pos_by_q)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(auc_by_question([pos_by_q[i] for i in idx], [neg_by_q[i] for i in idx]))
    stats.sort()
    return auc_by_question(pos_by_q, neg_by_q), stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def auc_diff_bootstrap(
    pos_a: Sequence[Sequence[float]], neg_a: Sequence[Sequence[float]],
    pos_b: Sequence[Sequence[float]], neg_b: Sequence[Sequence[float]],
    n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float, float]:
    """AUC(a) - AUC(b), resampling the same questions in both conditions."""
    n = len(pos_a)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(
            auc_by_question([pos_a[i] for i in idx], [neg_a[i] for i in idx])
            - auc_by_question([pos_b[i] for i in idx], [neg_b[i] for i in idx])
        )
    stats.sort()
    point = auc_by_question(pos_a, neg_a) - auc_by_question(pos_b, neg_b)
    return point, stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot)]


def percentile(values: Sequence[float], p: float) -> float:
    s = sorted(values)
    return s[min(int(p * len(s)), len(s) - 1)] if s else float("nan")
