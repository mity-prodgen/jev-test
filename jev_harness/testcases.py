"""Load batch test-case files (YAML or JSON) into Batch/Case objects.

A batch file looks like:

    batch: off_menu
    description: optional free text
    model: jev-latest        # optional, overridable with --model

    cases:
      - id: legal_complaint_01
        state: "..."
        tags: [off_menu]
        notes: optional free text
        questions:
          department:
            type: choice
            instructions: "Which team should handle this?"
            criteria:
              billing: "..."
              technical: "..."
              shipping: "..."
        expected:
          department: null   # no correct option exists for this case

`questions` is passed straight through to the TypeSafe API as the raw
question-dict form the SDK supports, so anything valid in the API reference
(criteria, structured instructions, extra fields) works here unmodified.
`expected` is harness-only bookkeeping and is never sent to the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Case:
    id: str
    state: Any
    questions: dict[str, dict[str, Any]]
    expected: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class Batch:
    name: str
    description: str | None
    model: str | None
    cases: list[Case]


def load_batch(path: str | Path) -> Batch:
    path = Path(path)
    text = path.read_text()
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)

    cases: list[Case] = []
    seen_ids: set[str] = set()
    for raw_case in data["cases"]:
        case_id = raw_case["id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id {case_id!r} in {path}")
        seen_ids.add(case_id)
        cases.append(
            Case(
                id=case_id,
                state=raw_case["state"],
                questions=raw_case["questions"],
                expected=raw_case.get("expected", {}),
                tags=raw_case.get("tags", []),
                notes=raw_case.get("notes"),
            )
        )

    return Batch(
        name=data.get("batch", path.stem),
        description=data.get("description"),
        model=data.get("model"),
        cases=cases,
    )
