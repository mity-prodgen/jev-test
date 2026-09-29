#!/usr/bin/env python3
"""Build a stratified sample of RAGTruth cases for the judge batch.

600 cases = 3 task_type (Summary/QA/Data2txt) x 2 classes (hallucinated/not)
x 100 each, from quality=='good' responses, fixed seed. Natural hallucination
rates by task range from 29% (QA) to 69% (Data2txt), so equal cells per
task/class - not proportional random sampling - is what gets us to "roughly
half hallucinated" while also giving clean per-task-type breakdowns later.

Not committed to git: writes to hallucination/results/judge_sample.jsonl,
which is gitignored (see scripts/download_ragtruth.sh for why). Rerun this
script to regenerate an identical sample.

Usage:
    python hallucination/build_sample.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260929
N_PER_CELL = 100  # per (task_type, hallucinated) cell -> 3 * 2 * 100 = 600
TASK_TYPES = ("Summary", "QA", "Data2txt")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ragtruth"
OUT_DIR = ROOT / "hallucination" / "results"


def load_dataset() -> tuple[list[dict], dict[str, dict]]:
    response_path = DATA_DIR / "response.jsonl"
    source_path = DATA_DIR / "source_info.jsonl"
    if not response_path.exists() or not source_path.exists():
        raise FileNotFoundError(
            f"RAGTruth files not found under {DATA_DIR}. Run scripts/download_ragtruth.sh first."
        )
    responses = [json.loads(line) for line in response_path.open()]
    sources = {}
    for line in source_path.open():
        s = json.loads(line)
        sources[s["source_id"]] = s
    return responses, sources


def build_cases(responses: list[dict], sources: dict[str, dict]) -> list[dict]:
    by_cell: dict[tuple[str, bool], list[dict]] = {}
    for r in responses:
        if r.get("quality") != "good":
            continue
        source = sources[r["source_id"]]
        task_type = source["task_type"]
        if task_type not in TASK_TYPES:
            continue
        hallucinated = len(r["labels"]) > 0
        case = {
            "response_id": r["id"],
            "source_id": r["source_id"],
            "task_type": task_type,
            "corpus": source["source"],
            "generating_model": r["model"],
            "source_passage": source["source_info"],
            "response": r["response"],
            "labels": r["labels"],
            "hallucinated": hallucinated,
        }
        by_cell.setdefault((task_type, hallucinated), []).append(case)

    rng = random.Random(SEED)
    sampled: list[dict] = []
    for task_type in TASK_TYPES:
        for hallucinated in (True, False):
            pool = by_cell.get((task_type, hallucinated), [])
            if len(pool) < N_PER_CELL:
                raise ValueError(
                    f"only {len(pool)} cases available for ({task_type}, hallucinated={hallucinated}), "
                    f"need {N_PER_CELL}"
                )
            sampled.extend(rng.sample(pool, N_PER_CELL))

    rng.shuffle(sampled)
    for i, case in enumerate(sampled):
        halluc_tag = "halluc" if case["hallucinated"] else "clean"
        case["case_id"] = f"{case['task_type'].lower()}_{halluc_tag}_{i:03d}"
    return sampled


def main() -> None:
    responses, sources = load_dataset()
    cases = build_cases(responses, sources)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "judge_sample.jsonl"
    with out_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    n_halluc = sum(1 for c in cases if c["hallucinated"])
    print(f"wrote {len(cases)} cases to {out_path} ({n_halluc} hallucinated, {len(cases) - n_halluc} clean)")
    for task_type in TASK_TYPES:
        n = sum(1 for c in cases if c["task_type"] == task_type)
        print(f"  {task_type}: {n}")


if __name__ == "__main__":
    main()
