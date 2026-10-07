"""Cached, threaded Jev requests for pair-scoring experiments (post 3).

Every request goes through DiskCache, so a rerun never re-calls the API. A
cache hit returns the latency that was measured on the original call.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .cache import DiskCache, cache_key
from .client import JevClient


@dataclass
class JevRequest:
    id: str
    state: Any
    questions: dict[str, dict[str, Any]]


def request_key(model: str, req: JevRequest) -> str:
    return cache_key("jev", model, req.state, req.questions)


def is_cached(cache: DiskCache, model: str, req: JevRequest) -> bool:
    return cache.get(request_key(model, req)) is not None


class _ClientPool:
    def __init__(self, model: str, max_retries: int, timeout: float) -> None:
        self.model, self.max_retries, self.timeout = model, max_retries, timeout
        self.local = threading.local()
        self.clients: list[JevClient] = []
        self.lock = threading.Lock()

    def get(self) -> JevClient:
        client = getattr(self.local, "client", None)
        if client is None:
            client = JevClient(model=self.model, max_retries=self.max_retries, timeout=self.timeout)
            client.__enter__()
            self.local.client = client
            with self.lock:
                self.clients.append(client)
        return client

    def close(self) -> None:
        for c in self.clients:
            c.__exit__(None, None, None)


def run_requests(
    requests: list[JevRequest],
    model: str,
    cache: DiskCache,
    max_workers: int = 12,
    max_retries: int = 6,
    timeout: float = 120.0,
    progress_every: int = 500,
) -> dict[str, dict[str, Any]]:
    """Returns {request id: {ok, model, raw_response, latency_ms, error, cached}}."""
    pool = _ClientPool(model, max_retries, timeout)
    results: dict[str, dict[str, Any]] = {}
    done = 0
    lock = threading.Lock()

    def one(req: JevRequest) -> tuple[str, dict[str, Any]]:
        nonlocal done

        def call() -> dict[str, Any]:
            r = pool.get().ask(req.state, req.questions)
            return {"ok": r.ok, "model": r.model, "raw_response": r.raw_response,
                    "latency_ms": r.latency_ms, "error": r.error}

        res, was_cached = cache.get_or_call(request_key(model, req), call)
        res = dict(res)
        res["cached"] = was_cached
        with lock:
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(requests)} requests done", flush=True)
        return req.id, res

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for rid, res in ex.map(one, requests):
                results[rid] = res
    finally:
        pool.close()
    return results


def answer(res: dict[str, Any], qid: str) -> dict[str, Any] | None:
    raw = res.get("raw_response") if res.get("ok") else None
    return (raw or {}).get("answers", {}).get(qid)


def noul_of(res: dict[str, Any], qid: str) -> float | None:
    a = answer(res, qid)
    return a.get("noul") if a else None


def score_of(res: dict[str, Any], qid: str) -> float | None:
    a = answer(res, qid)
    return a.get("score") if a else None


def input_tokens_of(res: dict[str, Any]) -> int | None:
    raw = res.get("raw_response") if res.get("ok") else None
    return ((raw or {}).get("usage") or {}).get("input_tokens")
