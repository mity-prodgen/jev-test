"""Token/cost estimation for --dry-run modes.

Neither TypeSafe nor Anthropic publish a tokenizer we can call directly, so
this uses tiktoken's cl100k_base as a documented proxy - an estimate, not an
exact count. Real per-call usage is always logged from the API's own `usage`
field once a call actually runs; this module only exists to answer "how much
will this cost" before spending anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import tiktoken

from .llm_judge import PRICING as LLM_JUDGE_PRICING

_ENCODING = tiktoken.get_encoding("cl100k_base")

# From docs.typesafe.ai/models: $42 per Btok input, output is free.
JEV_USD_PER_MTOK_INPUT = 0.042
JEV_USD_PER_MTOK_OUTPUT = 0.0


def estimate_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def estimate_state_tokens(state: Any) -> int:
    text = state if isinstance(state, str) else json.dumps(state)
    return estimate_tokens(text)


@dataclass
class StepEstimate:
    name: str
    n_calls: int
    est_input_tokens: int
    est_output_tokens: int
    usd_per_mtok_input: float
    usd_per_mtok_output: float

    @property
    def est_cost_usd(self) -> float:
        return (
            self.est_input_tokens * self.usd_per_mtok_input
            + self.est_output_tokens * self.usd_per_mtok_output
        ) / 1_000_000


def jev_step_estimate(name: str, texts: list[str], est_output_tokens_per_call: int = 20) -> StepEstimate:
    """texts: the full rendered state+instructions text for each planned call."""
    input_tokens = sum(estimate_tokens(t) for t in texts)
    return StepEstimate(
        name=name,
        n_calls=len(texts),
        est_input_tokens=input_tokens,
        est_output_tokens=est_output_tokens_per_call * len(texts),
        usd_per_mtok_input=JEV_USD_PER_MTOK_INPUT,
        usd_per_mtok_output=JEV_USD_PER_MTOK_OUTPUT,
    )


def llm_judge_step_estimate(
    name: str, texts: list[str], model: str, est_output_tokens_per_call: int = 20
) -> StepEstimate:
    in_price, out_price = LLM_JUDGE_PRICING.get(model, (0.0, 0.0))
    input_tokens = sum(estimate_tokens(t) for t in texts)
    return StepEstimate(
        name=name,
        n_calls=len(texts),
        est_input_tokens=input_tokens,
        est_output_tokens=est_output_tokens_per_call * len(texts),
        usd_per_mtok_input=in_price,
        usd_per_mtok_output=out_price,
    )


def print_dry_run(steps: list[StepEstimate]) -> None:
    print("=== DRY RUN: estimated calls and cost (no API calls made) ===")
    print(
        f"{'step':<24}{'calls':>8}{'est. input tok':>16}{'est. output tok':>17}{'est. cost':>12}"
    )
    total_calls = 0
    total_cost = 0.0
    for step in steps:
        print(
            f"{step.name:<24}{step.n_calls:>8}{step.est_input_tokens:>16,}"
            f"{step.est_output_tokens:>17,}{'$' + format(step.est_cost_usd, '.4f'):>12}"
        )
        total_calls += step.n_calls
        total_cost += step.est_cost_usd
    print("-" * 77)
    print(f"{'TOTAL':<24}{total_calls:>8}{'':>16}{'':>17}{'$' + format(total_cost, '.4f'):>12}")
    print()
    print(
        "Estimate uses tiktoken cl100k_base as a proxy tokenizer for both Jev and "
        "the LLM judge - actual tokenizers differ, so treat this as a ballpark, "
        "not an exact bill. Pass --live to actually run these calls."
    )
