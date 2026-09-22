"""Thin wrapper around typesafe-sdk that always returns the raw response body.

We go through result.raw_http_response.json() rather than the typed
SystemOneResponse so the harness logs exactly what the API sent, including any
answer kinds or fields the SDK version in requirements.txt doesn't know about
yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from typesafe_sdk import RetryPolicy, TypeSafeAPIError, TypeSafeClient


@dataclass
class RawResult:
    ok: bool
    model: str | None
    request_id: str | None
    latency_ms: float
    raw_response: dict[str, Any] | None
    error: str | None


class JevClient:
    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        retry = RetryPolicy(max_retries=max_retries, timeout=timeout)
        kwargs: dict[str, Any] = {"retry": retry}
        if model:
            kwargs["model"] = model
        self._client = TypeSafeClient(**kwargs)

    def __enter__(self) -> "JevClient":
        self._client.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.__exit__(*exc)

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> RawResult:
        start = monotonic()
        try:
            result = self._client.system_one(state, questions)
        except TypeSafeAPIError as error:
            return RawResult(
                ok=False,
                model=None,
                request_id=getattr(error, "request_id", None),
                latency_ms=(monotonic() - start) * 1000,
                raw_response=None,
                error=f"{getattr(error, 'status', '?')}: {error}",
            )

        raw = result.raw_http_response.json()
        return RawResult(
            ok=True,
            model=raw.get("model"),
            request_id=getattr(result, "request_id", None),
            latency_ms=(monotonic() - start) * 1000,
            raw_response=raw,
            error=None,
        )
