#!/usr/bin/env python3
"""Accuracy/precision/recall, calibration (ECE + reliability diagram),
isotonic recalibration + learning curve, and cost/latency for both judges.

Requires judge_results_jev.json and judge_results_llm_judge.json from
run_judge_batch.py --live. Writes CSV/JSON summaries, PNG charts, and
regenerates hallucination/RESULTS.md.

Usage:
    python hallucination/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jev_harness.calibration_analysis import (
    expected_calibration_error,
    learning_curve,
    split_train_test,
)
from jev_harness.stats import calibration_buckets

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "hallucination" / "results"
RESULTS_MD = ROOT / "hallucination" / "RESULTS.md"

RECAL_SEED = 20260929
N_TRAIN = 400
LEARNING_CURVE_SIZES = (50, 100, 200, 400)


def load_rows(judge: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"judge_results_{judge}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python hallucination/run_judge_batch.py --live")
    return json.loads(path.read_text())


def usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("error") is None and r.get("predicted_hallucinated") is not None]


def confusion_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp = sum(1 for r in rows if r["predicted_hallucinated"] and r["true_hallucinated"])
    fp = sum(1 for r in rows if r["predicted_hallucinated"] and not r["true_hallucinated"])
    fn = sum(1 for r in rows if not r["predicted_hallucinated"] and r["true_hallucinated"])
    tn = sum(1 for r in rows if not r["predicted_hallucinated"] and not r["true_hallucinated"])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float | None]:
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    accuracy = (c["tp"] + c["tn"]) / n if n else None
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
    return {"n": n, "accuracy": accuracy, "precision": precision, "recall": recall}


def breakdown_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | None]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(key) or "unknown", []).append(r)
    return {g: metrics_from_confusion(confusion_counts(rs)) for g, rs in groups.items()}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(int(p * len(s)), len(s) - 1)
    return s[idx]


def cost_latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    return {
        "n": len(rows),
        "median_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_cost_usd": (sum(costs) / len(costs)) if costs else None,
        "cost_per_1000_usd": (sum(costs) / len(costs) * 1000) if costs else None,
        "total_cost_usd": sum(costs) if costs else None,
    }


def calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Standardize to stats.py's {confidence, correct} convention."""
    out = []
    for r in rows:
        if r.get("judge_confidence") is None:
            continue
        out.append(
            {
                "confidence": r["judge_confidence"],
                "correct": r["predicted_hallucinated"] == r["true_hallucinated"],
            }
        )
    return out


def plot_reliability_diagram(buckets: list[dict[str, Any]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    xs = [b["mean_confidence"] for b in buckets]
    ys = [b["empirical_accuracy"] for b in buckets]
    ns = [b["n"] for b in buckets]
    sizes = [max(30, n * 3) for n in ns]
    ax.scatter(xs, ys, s=sizes, alpha=0.75, color="#2a78d6", label="observed buckets")
    for b in buckets:
        ax.annotate(f"n={b['n']}", (b["mean_confidence"], b["empirical_accuracy"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_type_accuracy(jev_breakdown: dict, llm_breakdown: dict, path: Path) -> None:
    types = sorted(set(jev_breakdown) | set(llm_breakdown))
    x = range(len(types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    jev_vals = [jev_breakdown.get(t, {}).get("accuracy") or 0 for t in types]
    llm_vals = [llm_breakdown.get(t, {}).get("accuracy") or 0 for t in types]
    ax.bar([i - width / 2 for i in x], jev_vals, width, label="Jev", color="#2a78d6")
    ax.bar([i + width / 2 for i in x], llm_vals, width, label="LLM judge", color="#eb6834")
    ax.set_xticks(list(x))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by error type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(curve: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [c["n_train"] for c in curve]
    before = curve[0]["ece_before"] if curve else None
    after_vals = [c["ece_after"] for c in curve]
    ax.plot(ns, after_vals, marker="o", color="#2a78d6", label="ECE after recalibration")
    if before is not None:
        ax.axhline(before, linestyle="--", color="gray", label="ECE before (no recalibration)")
    ax.set_xlabel("Training labels used")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Recalibration: ECE vs. training set size (Jev)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_results_md(summary: dict[str, Any]) -> None:
    jev = summary["jev"]
    llm = summary["llm_judge"]
    lines = [
        "# Results: Can Jev catch hallucinations?",
        "",
        f"Regenerated by `analyze.py`. Sample: {jev['metrics']['n']} judge cases from RAGTruth "
        "(600 target; see `hallucination/build_sample.py`).",
        "",
        "## Accuracy, precision, recall (hallucinated class)",
        "",
        "| Judge | n | Accuracy | Precision | Recall |",
        "|---|---|---|---|---|",
        f"| Jev ({jev['model']}) | {jev['metrics']['n']} | "
        f"{_pct(jev['metrics']['accuracy'])} | {_pct(jev['metrics']['precision'])} | {_pct(jev['metrics']['recall'])} |",
        f"| LLM judge ({llm['model']}) | {llm['metrics']['n']} | "
        f"{_pct(llm['metrics']['accuracy'])} | {_pct(llm['metrics']['precision'])} | {_pct(llm['metrics']['recall'])} |",
        "",
        "## Calibration (Jev)",
        "",
        f"- ECE (raw): {jev['ece']:.4f}" if jev["ece"] is not None else "- ECE (raw): n/a",
        f"- ECE after isotonic recalibration ({N_TRAIN}-case train / {jev['recal']['test_n']}-case test): "
        f"{jev['recal']['ece_before']:.4f} -> {jev['recal']['ece_after']:.4f}"
        if jev["recal"]
        else "- recalibration: n/a",
        "",
        "## Cost & latency",
        "",
        "| Judge | median latency (ms) | p95 latency (ms) | cost / 1,000 judgments |",
        "|---|---|---|---|",
        f"| Jev | {_num(jev['cost_latency']['median_latency_ms'])} | "
        f"{_num(jev['cost_latency']['p95_latency_ms'])} | "
        f"{_usd(jev['cost_latency']['cost_per_1000_usd'])} |",
        f"| LLM judge | {_num(llm['cost_latency']['median_latency_ms'])} | "
        f"{_num(llm['cost_latency']['p95_latency_ms'])} | "
        f"{_usd(llm['cost_latency']['cost_per_1000_usd'])} |",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))


def _pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "n/a"


def _num(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "n/a"


def _usd(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "n/a"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for judge_key in ("jev", "llm_judge"):
        raw_rows = load_rows(judge_key)
        rows = usable_rows(raw_rows)
        n_excluded = len(raw_rows) - len(rows)

        metrics = metrics_from_confusion(confusion_counts(rows))
        by_error_type = breakdown_by(rows, "error_type")
        by_task_type = breakdown_by(rows, "task_type")
        cost_latency = cost_latency_summary(raw_rows)
        model = next((r["model"] for r in raw_rows if r.get("model")), None)

        cal_rows = calibration_rows(rows)
        buckets = calibration_buckets(cal_rows)
        ece = expected_calibration_error(cal_rows) if cal_rows else None
        plot_reliability_diagram(
            buckets, RESULTS_DIR / f"reliability_{judge_key}.png", f"Reliability diagram: {judge_key}"
        )

        recal_summary = None
        if judge_key == "jev" and len(cal_rows) >= N_TRAIN + 50:
            train_rows, test_rows = split_train_test(cal_rows, N_TRAIN, seed=RECAL_SEED)
            curve = learning_curve(train_rows, test_rows, sizes=LEARNING_CURVE_SIZES)
            plot_learning_curve(curve, RESULTS_DIR / "recalibration_learning_curve.png")
            recal_summary = {
                "test_n": len(test_rows),
                "ece_before": curve[-1]["ece_before"] if curve else None,
                "ece_after": curve[-1]["ece_after"] if curve else None,
                "learning_curve": curve,
            }

        summary[judge_key] = {
            "model": model,
            "n_excluded_errors": n_excluded,
            "metrics": metrics,
            "by_error_type": by_error_type,
            "by_task_type": by_task_type,
            "cost_latency": cost_latency,
            "ece": ece,
            "recal": recal_summary,
        }

        print(f"\n=== {judge_key} ({model}) ===")
        print(f"  n={metrics['n']} (excluded {n_excluded} error rows)")
        print(f"  accuracy={_pct(metrics['accuracy'])} precision={_pct(metrics['precision'])} recall={_pct(metrics['recall'])}")
        print(f"  ECE={ece:.4f}" if ece is not None else "  ECE=n/a")
        print(f"  cost/1000={_usd(cost_latency['cost_per_1000_usd'])} median_latency={_num(cost_latency['median_latency_ms'])}ms")

    plot_per_type_accuracy(summary["jev"]["by_error_type"], summary["llm_judge"]["by_error_type"], RESULTS_DIR / "accuracy_by_error_type.png")

    (RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    write_results_md(summary)
    print(f"\nwrote {RESULTS_DIR / 'analysis_summary.json'}, PNG charts under {RESULTS_DIR}, and {RESULTS_MD}")


if __name__ == "__main__":
    main()
