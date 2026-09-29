#!/usr/bin/env python3
"""Run both judges (Jev + LLM judge) over the sampled RAGTruth cases.

Default is --dry-run: prints call-count/token/cost estimates and makes zero
API calls. Pass --live to actually run. Every raw response is cached to disk
(jev_harness.cache.DiskCache) so a rerun of this script costs nothing.

Usage:
    python hallucination/run_judge_batch.py --dry-run
    python hallucination/run_judge_batch.py --live --limit 10      # smoke test
    python hallucination/run_judge_batch.py --live                 # full 600-case run
    python hallucination/run_judge_batch.py --live --judge jev     # one judge only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from jev_harness.cache import DiskCache, cache_key
from jev_harness.client import JevClient
from jev_harness.cost import jev_step_estimate, llm_judge_step_estimate, print_dry_run
from jev_harness.llm_judge import LLMJudgeClient

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "hallucination" / "results"
CACHE_DIR = ROOT / "cache"

JEV_MODEL_DEFAULT = "jev-1.13.0"

JUDGE_INSTRUCTIONS = (
    "Every factual claim in `response` is directly supported by `source_passage`. "
    "`true` requires every claim in `response` to trace to an explicit statement in "
    "`source_passage`, with nothing added or contradicted. `false` if any claim in "
    "`response` contradicts `source_passage`, states a number, date, or name not "
    "present in `source_passage`, or adds a specific detail `source_passage` does not support."
)
JUDGE_CRITERIA = {
    "true": "Every claim in `response` can be traced to an explicit statement in `source_passage`.",
    "false": (
        "At least one claim in `response` contradicts `source_passage`, or states/adds "
        "a specific detail not present in it."
    ),
}


def load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    path = RESULTS_DIR / "judge_sample.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run build_sample.py (then error_types.py) first")
    cases = [json.loads(line) for line in path.open()]
    return cases[:limit] if limit else cases


def jev_state(case: dict[str, Any]) -> dict[str, Any]:
    return {"source_passage": case["source_passage"], "response": case["response"]}


def _jev_render_text(case: dict[str, Any]) -> str:
    """Flat text for --dry-run token estimation only - not the real request shape."""
    return json.dumps(jev_state(case)) + JUDGE_INSTRUCTIONS + json.dumps(JUDGE_CRITERIA)


def _llm_render_text(case: dict[str, Any]) -> str:
    passage = case["source_passage"]
    passage_text = passage if isinstance(passage, str) else json.dumps(passage)
    return passage_text + case["response"]


def run_jev(cases: list[dict[str, Any]], cache: DiskCache, model: str) -> list[dict[str, Any]]:
    rows = []
    with JevClient(model=model) as client:
        for case in cases:
            state = jev_state(case)
            questions = {
                "supported": {
                    "type": "noul",
                    "instructions": JUDGE_INSTRUCTIONS,
                    "criteria": JUDGE_CRITERIA,
                }
            }
            key = cache_key("jev", model, state, questions)

            def _call(state=state, questions=questions) -> dict[str, Any]:
                result = client.ask(state, questions)
                return {
                    "ok": result.ok,
                    "model": result.model,
                    "raw_response": result.raw_response,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }

            result, was_cached = cache.get_or_call(key, _call)
            rows.append(_jev_row(case, result, was_cached))
    return rows


def _jev_row(case: dict[str, Any], result: dict[str, Any], was_cached: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "task_type": case["task_type"],
        "error_type": case.get("error_type"),
        "true_hallucinated": case["hallucinated"],
        "judge": "jev",
        "model": result.get("model"),
        "cached": was_cached,
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
        "p_hallucinated": None,
        "predicted_hallucinated": None,
        "judge_confidence": None,
        "input_tokens": None,
        "output_tokens": 0,
        "cost_usd": None,
    }
    if result.get("ok") and result.get("raw_response"):
        answer = result["raw_response"].get("answers", {}).get("supported", {})
        noul = answer.get("noul")  # P(the "supported" question is true) = P(NOT hallucinated)
        if noul is not None:
            p_hallucinated = 1 - noul
            row["p_hallucinated"] = p_hallucinated
            row["predicted_hallucinated"] = p_hallucinated >= 0.5
            row["judge_confidence"] = max(p_hallucinated, 1 - p_hallucinated)
        row["input_tokens"] = result["raw_response"].get("usage", {}).get("input_tokens")
        if row["input_tokens"] is not None:
            row["cost_usd"] = row["input_tokens"] * 0.042 / 1_000_000
    return row


def run_llm_judge(
    cases: list[dict[str, Any]], cache: DiskCache, judge_client: LLMJudgeClient
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        key = cache_key("llm_judge", judge_client.model, case["source_passage"], case["response"])

        def _call(case=case) -> dict[str, Any]:
            result = judge_client.ask(case["source_passage"], case["response"])
            return {
                "ok": result.ok,
                "hallucinated": result.hallucinated,
                "confidence": result.confidence,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "raw_response": result.raw_response,
            }

        result, was_cached = cache.get_or_call(key, _call)
        rows.append(_llm_judge_row(case, result, was_cached))
    return rows


def _llm_judge_row(case: dict[str, Any], result: dict[str, Any], was_cached: bool) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "task_type": case["task_type"],
        "error_type": case.get("error_type"),
        "true_hallucinated": case["hallucinated"],
        "judge": "llm_judge",
        "model": result.get("model"),
        "cached": was_cached,
        "latency_ms": result.get("latency_ms"),
        "error": result.get("error"),
        "p_hallucinated": None,
        "predicted_hallucinated": result.get("hallucinated"),
        "judge_confidence": result.get("confidence"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "cost_usd": result.get("cost_usd"),
    }


def _write_rows(rows: list[dict[str, Any]], suffix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"judge_results_{suffix}.json").write_text(json.dumps(rows, indent=2))
    if rows:
        with (RESULTS_DIR / f"judge_results_{suffix}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="actually call the APIs (default: dry-run)")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N cases (smoke testing)")
    parser.add_argument("--judge", choices=["jev", "llm_judge", "both"], default="both")
    parser.add_argument("--jev-model", default=JEV_MODEL_DEFAULT)
    args = parser.parse_args()

    cases = load_cases(limit=args.limit)
    run_jev_flag = args.judge in ("jev", "both")
    run_llm_flag = args.judge in ("llm_judge", "both")

    if not args.live:
        steps = []
        if run_jev_flag:
            steps.append(jev_step_estimate("jev", [_jev_render_text(c) for c in cases]))
        if run_llm_flag:
            from os import environ

            model = environ.get("LLM_JUDGE_MODEL", "claude-haiku-4-5")
            steps.append(llm_judge_step_estimate("llm_judge", [_llm_render_text(c) for c in cases], model))
        print_dry_run(steps)
        return

    cache = DiskCache(CACHE_DIR, "judge_batch")

    if run_jev_flag:
        jev_rows = run_jev(cases, cache, model=args.jev_model)
        _write_rows(jev_rows, "jev")
        n_cached = sum(1 for r in jev_rows if r["cached"])
        n_errors = sum(1 for r in jev_rows if r["error"])
        print(f"jev: {len(jev_rows)} rows ({n_cached} from cache, {n_errors} errors)")

    if run_llm_flag:
        judge_client = LLMJudgeClient()
        llm_rows = run_llm_judge(cases, cache, judge_client)
        _write_rows(llm_rows, "llm_judge")
        n_cached = sum(1 for r in llm_rows if r["cached"])
        n_errors = sum(1 for r in llm_rows if r["error"])
        print(f"llm_judge ({judge_client.model}): {len(llm_rows)} rows ({n_cached} from cache, {n_errors} errors)")


if __name__ == "__main__":
    main()
