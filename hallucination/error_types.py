#!/usr/bin/env python3
"""Heuristic error-type tagging over the judge sample.

This is explicitly a heuristic, not ground truth - RAGTruth gives span-level
label_type (Evident/Subtle x Conflict/Baseless Info) but nothing finer. We derive:
  - supported: no labels (not hallucinated)
  - unsupported_extra_detail: primary span's label_type is *Baseless Info*
  - numeric_or_date_error / negation_or_contradiction / wrong_entity:
    sub-classified from *Conflict* spans by regex over the span text

A response with multiple spans takes the most-severe (Evident before Subtle),
first-listed span as its single tag. Also writes a 30-case random spot-check
CSV (span text + assigned type) - review it before trusting the per-error-type
accuracy breakdown in analyze.py.

Usage:
    python hallucination/error_types.py   # requires judge_sample.jsonl (run build_sample.py first)
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "hallucination" / "results"
SPOTCHECK_SEED = 20260929
SPOTCHECK_N = 30

_NUMERIC_DATE_RE = re.compile(
    r"\$?\d[\d,.]*%?|"
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|no|never|none|without|n't|neither|nor|nothing|nobody|nowhere)\b",
    re.IGNORECASE,
)


def _severity(label_type: str) -> int:
    return 0 if label_type.startswith("Evident") else 1


def classify_span(label_type: str, span_text: str) -> str:
    if "Baseless Info" in label_type:
        return "unsupported_extra_detail"
    if _NUMERIC_DATE_RE.search(span_text):
        return "numeric_or_date_error"
    if _NEGATION_RE.search(span_text):
        return "negation_or_contradiction"
    return "wrong_entity"


def tag_case(case: dict) -> str:
    if not case["labels"]:
        return "supported"
    ordered = sorted(case["labels"], key=lambda lab: _severity(lab.get("label_type", "")))
    primary = ordered[0]
    return classify_span(primary.get("label_type", ""), primary.get("text", ""))


def main() -> None:
    sample_path = RESULTS_DIR / "judge_sample.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(f"{sample_path} not found - run build_sample.py first")

    cases = [json.loads(line) for line in sample_path.open()]
    for case in cases:
        case["error_type"] = tag_case(case)

    with sample_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    counts = Counter(c["error_type"] for c in cases)
    print("error_type distribution:")
    for etype, n in counts.most_common():
        print(f"  {etype}: {n}")

    halluc_cases = [c for c in cases if c["labels"]]
    rng = random.Random(SPOTCHECK_SEED)
    spot = rng.sample(halluc_cases, min(SPOTCHECK_N, len(halluc_cases)))

    spotcheck_path = RESULTS_DIR / "error_type_spotcheck.csv"
    with spotcheck_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case_id", "task_type", "assigned_error_type", "primary_label_type", "primary_span_text"]
        )
        for c in spot:
            ordered = sorted(c["labels"], key=lambda lab: _severity(lab.get("label_type", "")))
            primary = ordered[0]
            writer.writerow(
                [
                    c["case_id"],
                    c["task_type"],
                    c["error_type"],
                    primary.get("label_type", ""),
                    primary.get("text", ""),
                ]
            )
    print(f"\nwrote {len(spot)}-case spot-check to {spotcheck_path} - review before trusting the per-type breakdown")


if __name__ == "__main__":
    main()
