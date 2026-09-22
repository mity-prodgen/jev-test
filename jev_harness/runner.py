"""Run a batch of test cases against Jev, log raw responses, export results."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .client import JevClient
from .stats import grade, summarize
from .testcases import Case, load_batch


def run_batch(
    batch_path: str,
    out_dir: str = "results",
    model: str | None = None,
    max_retries: int = 3,
) -> Path:
    batch = load_batch(batch_path)
    run_model = model or batch.model

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(out_dir) / batch.name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    raw_log_path = run_dir / "raw_responses.jsonl"

    with JevClient(model=run_model, max_retries=max_retries) as client:
        with raw_log_path.open("w") as raw_log:
            for case in batch.cases:
                result = client.ask(case.state, case.questions)

                raw_log.write(
                    json.dumps(
                        {
                            "case_id": case.id,
                            "state": case.state,
                            "questions": case.questions,
                            "tags": case.tags,
                            "ok": result.ok,
                            "error": result.error,
                            "request_id": result.request_id,
                            "latency_ms": result.latency_ms,
                            "raw_response": result.raw_response,
                        }
                    )
                    + "\n"
                )

                rows.extend(_rows_for_case(case, result.ok, result.model, result.request_id, result.latency_ms, result.raw_response, result.error))

    _write_csv(run_dir / "results.csv", rows)
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2))

    summary = summarize(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return run_dir


def _rows_for_case(
    case: Case,
    ok: bool,
    model: str | None,
    request_id: str | None,
    latency_ms: float,
    raw_response: dict[str, Any] | None,
    error: str | None,
) -> list[dict[str, Any]]:
    if not ok:
        return [
            {
                "case_id": case.id,
                "tags": ",".join(case.tags),
                "question_id": None,
                "type": None,
                "value": None,
                "confidence": None,
                "probabilities": None,
                "expected": None,
                "correct": None,
                "model": None,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 1),
                "error": error,
            }
        ]

    answers = (raw_response or {}).get("answers", {})
    rows = []
    for qid, question in case.questions.items():
        answer = answers.get(qid, {})
        qtype = answer.get("type", question.get("type"))

        if qtype == "choice":
            value = answer.get("choice")
            confidence = answer.get("confidence")
        elif qtype == "score":
            value = answer.get("score")
            confidence = answer.get("confidence")
        elif qtype == "noul":
            value = answer.get("noul")
            confidence = None
        else:
            value = None
            confidence = None

        expected = case.expected.get(qid)
        correct = grade(qtype, value, expected) if qtype else None

        probabilities = answer.get("probabilities")
        rows.append(
            {
                "case_id": case.id,
                "tags": ",".join(case.tags),
                "question_id": qid,
                "type": qtype,
                "value": value,
                "confidence": confidence,
                "probabilities": json.dumps(probabilities) if probabilities else None,
                "expected": expected,
                "correct": correct,
                "model": model,
                "request_id": request_id,
                "latency_ms": round(latency_ms, 1),
                "error": None,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
