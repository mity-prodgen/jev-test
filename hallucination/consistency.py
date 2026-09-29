#!/usr/bin/env python3
"""Step 5: run-to-run consistency.

50 cases spread across Jev's observed p range (from run_judge_batch.py's Jev
results), with >=15 forced into [0.4, 0.6], each sent 20 independent times
(cache bypassed via a repeat_index folded into the cache key). Reports the
spread of p per case, and whether spread is larger near 0.5 - the failure
mode a commenter raised on post 1 about Jev's confidence specifically.

Jev only by default, since that's what the comment was about; --include-llm-
judge would be the natural extension but is not wired up here - see the note
in main().

Usage:
    python hallucination/consistency.py --dry-run
    python hallucination/consistency.py --live --n-cases 3 --reps 3   # mini smoke test
    python hallucination/consistency.py --live                        # full 50 x 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jev_harness.cache import DiskCache, cache_key
from jev_harness.client import JevClient
from jev_harness.cost import jev_step_estimate, print_dry_run
from run_judge_batch import JEV_MODEL_DEFAULT, JUDGE_CRITERIA, JUDGE_INSTRUCTIONS, jev_state

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "hallucination" / "results"
CACHE_DIR = ROOT / "cache"

SEED = 20260929
N_CASES_DEFAULT = 50
N_NEAR_HALF_DEFAULT = 15
N_REPS_DEFAULT = 20
NEAR_HALF_RANGE = (0.4, 0.6)


def load_jev_results() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "judge_results_jev.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python hallucination/run_judge_batch.py --live")
    return json.loads(path.read_text())


def load_cases_by_id() -> dict[str, dict[str, Any]]:
    path = RESULTS_DIR / "judge_sample.jsonl"
    return {(c := json.loads(line))["case_id"]: c for line in path.open()}


def select_cases(
    jev_rows: list[dict[str, Any]], n_cases: int, n_near_half: int, seed: int
) -> list[dict[str, Any]]:
    usable = [r for r in jev_rows if r.get("p_hallucinated") is not None]
    near_half = [r for r in usable if NEAR_HALF_RANGE[0] <= r["p_hallucinated"] <= NEAR_HALF_RANGE[1]]
    near_half_ids = {r["case_id"] for r in near_half}
    rest = [r for r in usable if r["case_id"] not in near_half_ids]

    rng = random.Random(seed)
    n_near_half = min(n_near_half, len(near_half))
    picked_near = rng.sample(near_half, n_near_half)

    rest_sorted = sorted(rest, key=lambda r: r["p_hallucinated"])
    n_rest = max(0, n_cases - n_near_half)
    picked_rest: list[dict[str, Any]] = []
    if n_rest > 0 and rest_sorted:
        step = max(1, len(rest_sorted) // n_rest)
        picked_rest = [rest_sorted[min(i * step, len(rest_sorted) - 1)] for i in range(n_rest)]

    return (picked_near + picked_rest)[:n_cases]


def run_jev_repeats(
    cases: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    cache: DiskCache,
    model: str,
    n_reps: int,
) -> list[dict[str, Any]]:
    rows = []
    with JevClient(model=model) as client:
        for case_row in cases:
            case = cases_by_id[case_row["case_id"]]
            state = jev_state(case)
            questions = {
                "supported": {"type": "noul", "instructions": JUDGE_INSTRUCTIONS, "criteria": JUDGE_CRITERIA}
            }
            for rep in range(n_reps):
                key = cache_key("consistency_jev", model, state, questions, rep)

                def _call(state=state, questions=questions) -> dict[str, Any]:
                    result = client.ask(state, questions)
                    return {"ok": result.ok, "raw_response": result.raw_response, "error": result.error}

                result, _ = cache.get_or_call(key, _call)
                p = None
                if result.get("ok") and result.get("raw_response"):
                    noul = result["raw_response"].get("answers", {}).get("supported", {}).get("noul")
                    if noul is not None:
                        p = 1 - noul  # P(the "supported" question is true) = P(NOT hallucinated)
                rows.append(
                    {"case_id": case["case_id"], "repeat_index": rep, "p_hallucinated": p, "error": result.get("error")}
                )
    return rows


def summarize_spread(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[float]] = {}
    for r in rows:
        if r["p_hallucinated"] is not None:
            by_case.setdefault(r["case_id"], []).append(r["p_hallucinated"])
    out = []
    for case_id, values in by_case.items():
        mean_p = statistics.mean(values)
        out.append(
            {
                "case_id": case_id,
                "n": len(values),
                "mean_p": mean_p,
                "stdev_p": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "range_p": max(values) - min(values),
                "dist_from_half": abs(mean_p - 0.5),
            }
        )
    return out


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def plot_spread(summary_rows: list[dict[str, Any]], path: Path) -> None:
    summary_rows = sorted(summary_rows, key=lambda r: r["mean_p"])
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(summary_rows)))
    means = [r["mean_p"] for r in summary_rows]
    stdevs = [r["stdev_p"] for r in summary_rows]
    ax.errorbar(xs, means, yerr=stdevs, fmt="o", color="#2a78d6", ecolor="#9ec5f4", capsize=3, markersize=4)
    ax.axhline(0.5, linestyle="--", color="gray", label="p=0.5")
    ax.set_xlabel("Case (sorted by mean p)")
    ax.set_ylabel("p_hallucinated (mean +/- stdev across reps)")
    ax.set_title("Run-to-run consistency: spread per case")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


RESULTS_MD = ROOT / "hallucination" / "RESULTS.md"
CONSISTENCY_MARKER = "## Consistency (run-to-run)"


def write_consistency_results_md(
    summary_rows: list[dict[str, Any]], corr: float | None, reps: int
) -> None:
    import statistics as stats_mod

    mean_stdev = stats_mod.mean(r["stdev_p"] for r in summary_rows)
    near_half = [r for r in summary_rows if r["dist_from_half"] < 0.05]
    extreme = [r for r in summary_rows if r["dist_from_half"] > 0.45]
    section = [
        CONSISTENCY_MARKER,
        "",
        f"{len(summary_rows)} cases, {reps} independent calls each (cache bypassed).",
        "",
        f"- Mean stdev of `p_hallucinated` across reps: {mean_stdev:.4f}",
        f"- Correlation(distance from 0.5, stdev): {corr:.3f}" if corr is not None else "- Correlation: n/a",
    ]
    if near_half:
        section.append(
            f"- Mean stdev for cases within 0.05 of p=0.5 (n={len(near_half)}): "
            f"{stats_mod.mean(r['stdev_p'] for r in near_half):.4f}"
        )
    if extreme:
        section.append(
            f"- Mean stdev for cases >0.45 from p=0.5 (n={len(extreme)}): "
            f"{stats_mod.mean(r['stdev_p'] for r in extreme):.4f}"
        )
    section.append("")
    new_section_text = "\n".join(section)

    existing = RESULTS_MD.read_text() if RESULTS_MD.exists() else "# Results: Can Jev catch hallucinations?\n\n"
    if CONSISTENCY_MARKER in existing:
        before, _, _after_and_rest = existing.partition(CONSISTENCY_MARKER)
        # drop everything from the marker to the next top-level heading (or EOF)
        rest = existing.split(CONSISTENCY_MARKER, 1)[1]
        next_heading = rest.find("\n## ", 1)
        tail = rest[next_heading:] if next_heading != -1 else ""
        updated = before + new_section_text + tail
    else:
        updated = existing.rstrip("\n") + "\n\n" + new_section_text
    RESULTS_MD.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="actually call the API (default: dry-run)")
    parser.add_argument("--n-cases", type=int, default=N_CASES_DEFAULT)
    parser.add_argument("--n-near-half", type=int, default=N_NEAR_HALF_DEFAULT)
    parser.add_argument("--reps", type=int, default=N_REPS_DEFAULT)
    parser.add_argument("--jev-model", default=JEV_MODEL_DEFAULT)
    args = parser.parse_args()

    jev_rows = load_jev_results()
    cases_by_id = load_cases_by_id()
    selected = select_cases(jev_rows, args.n_cases, args.n_near_half, seed=SEED)

    if not args.live:
        texts = []
        for row in selected:
            case = cases_by_id[row["case_id"]]
            text = json.dumps({"source_passage": case["source_passage"], "response": case["response"]})
            texts.extend([text] * args.reps)
        print_dry_run([jev_step_estimate("consistency (jev)", texts)])
        return

    cache = DiskCache(CACHE_DIR, "consistency")
    rows = run_jev_repeats(selected, cases_by_id, cache, args.jev_model, args.reps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "consistency_raw.json").write_text(json.dumps(rows, indent=2))
    summary_rows = summarize_spread(rows)
    if not summary_rows:
        print("no usable rows - all calls errored?")
        return
    with (RESULTS_DIR / "consistency_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_spread(summary_rows, RESULTS_DIR / "consistency_spread.png")

    corr = correlation([r["dist_from_half"] for r in summary_rows], [r["stdev_p"] for r in summary_rows])
    write_consistency_results_md(summary_rows, corr, args.reps)
    print(f"cases={len(summary_rows)} reps={args.reps}")
    print(f"mean stdev: {statistics.mean(r['stdev_p'] for r in summary_rows):.4f}")
    print(f"correlation(|p-0.5|, stdev): {corr:.3f}" if corr is not None else "correlation: n/a")
    print(f"wrote {RESULTS_DIR / 'consistency_summary.csv'} and {RESULTS_DIR / 'consistency_spread.png'}")


if __name__ == "__main__":
    main()
