"""LLM judge client used as the comparison baseline against JevClient.

Provider-agnostic by design (LLM_JUDGE_PROVIDER / LLM_JUDGE_MODEL env vars),
but only the "anthropic" backend is implemented - add an adapter here for a
second provider without touching call sites in hallucination/run_judge_batch.py.

Uses a plain-text-prompted JSON response rather than the Messages API's
output_config structured-output feature, since this harness hasn't verified
that feature's exact request shape against the current SDK - a prompted JSON
object plus a regex/json.loads parse is simple, well-documented, and gives the
same {hallucinated, confidence} shape with an explicit, loggable parse_error
on the (rare) malformed response instead of a silent SDK-shape mismatch.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any

JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checking judge. You will be given a source passage "
    "and a response. Decide whether every factual claim in the response is "
    "directly supported by the source passage. A claim is supported only if "
    "the source states it explicitly or it follows with no added assumptions. "
    "Answer with hallucinated=true if any claim contradicts the source, states "
    "a number, date, or name not present in the source, or adds a specific "
    "detail the source does not support.\n\n"
    "Your entire reply must be exactly one line matching this pattern and "
    'nothing else: {"hallucinated": true|false, "confidence": <integer 0-100>}. '
    '"confidence" is how certain you are in that verdict, from 0 (a coin flip) '
    "to 100 (certain). Do not explain your reasoning. Do not list issues. Do "
    "not add any text, markdown, or code fences before or after the JSON - "
    "the JSON object is the complete response, first character to last."
)

# $ per million tokens: (input, output). Extend for other provider/model combos.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeResult:
    ok: bool
    hallucinated: bool | None
    confidence: float | None  # normalized to 0-1
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float
    raw_response: dict[str, Any] | None
    error: str | None


def _parse_judgment(text: str) -> tuple[bool, float]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in judge output: {text!r}")
    data = json.loads(match.group(0))
    return bool(data["hallucinated"]), float(data["confidence"]) / 100.0


class LLMJudgeClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 200,
    ) -> None:
        self.provider = provider or os.environ.get("LLM_JUDGE_PROVIDER", "anthropic")
        self.model = model or os.environ.get("LLM_JUDGE_MODEL", "claude-haiku-4-5")
        self.max_tokens = max_tokens
        if self.provider != "anthropic":
            raise NotImplementedError(
                f"LLM_JUDGE_PROVIDER={self.provider!r} has no adapter yet; only "
                "'anthropic' is implemented. Add a branch in LLMJudgeClient.__init__/.ask()."
            )
        import anthropic

        self._client = anthropic.Anthropic()

    def ask(self, source_passage: Any, response_text: str) -> JudgeResult:
        if not isinstance(source_passage, str):
            source_passage = json.dumps(source_passage, indent=2)
        user_content = f"SOURCE PASSAGE:\n{source_passage}\n\nRESPONSE:\n{response_text}"

        start = monotonic()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as error:  # noqa: BLE001 - logged like JevClient.ask(), not raised
            return JudgeResult(
                ok=False,
                hallucinated=None,
                confidence=None,
                model=None,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                latency_ms=(monotonic() - start) * 1000,
                raw_response=None,
                error=f"{type(error).__name__}: {error}",
            )

        latency_ms = (monotonic() - start) * 1000
        text = next((b.text for b in message.content if b.type == "text"), "")
        raw = message.to_dict()
        cost = self._cost(message.usage.input_tokens, message.usage.output_tokens)

        try:
            hallucinated, confidence = _parse_judgment(text)
        except (ValueError, KeyError, TypeError) as error:
            return JudgeResult(
                ok=False,
                hallucinated=None,
                confidence=None,
                model=message.model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                raw_response=raw,
                error=f"parse_error: {error}",
            )

        return JudgeResult(
            ok=True,
            hallucinated=hallucinated,
            confidence=confidence,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            raw_response=raw,
            error=None,
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float | None:
        pricing = PRICING.get(self.model)
        if pricing is None:
            return None
        in_price, out_price = pricing
        return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
