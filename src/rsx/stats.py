"""Progress logging during a run and summaries of a run directory."""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .results import STATUS_OK, Record

log = logging.getLogger(__name__)


class ProgressTracker:
    """Called for every record; logs throughput and ETA at most every `every_s` seconds."""

    def __init__(self, total: int, every_s: float = 30.0):
        self.total = total
        self.every_s = every_s
        self.t0 = time.monotonic()
        self._last = self.t0
        self.done = 0
        self.ok = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def update(self, rec: Record) -> None:
        self.done += 1
        self.ok += rec.status == STATUS_OK
        if rec.usage:
            self.prompt_tokens += rec.usage.get("prompt_tokens") or 0
            self.completion_tokens += rec.usage.get("completion_tokens") or 0
        now = time.monotonic()
        if now - self._last >= self.every_s or self.done == self.total:
            self._last = now
            self.log(now)

    def log(self, now: float | None = None) -> None:
        elapsed = max((now or time.monotonic()) - self.t0, 1e-6)
        rate = self.done / elapsed
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        log.info(
            "progress %d/%d (ok %d) | %.0f reports/h | %.0f prompt tok/s | %.0f gen tok/s | ETA %s",
            self.done,
            self.total,
            self.ok,
            rate * 3600,
            self.prompt_tokens / elapsed,
            self.completion_tokens / elapsed,
            _fmt(eta),
        )

    def summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.t0
        return {
            "done": self.done,
            "ok": self.ok,
            "elapsed_s": round(elapsed, 1),
            "reports_per_hour": round(self.done / elapsed * 3600, 1) if elapsed > 0 else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "gen_tokens_per_s": round(self.completion_tokens / elapsed, 1) if elapsed > 0 else None,
        }


def _fmt(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def summarize(records: Iterable[Record]) -> dict[str, Any]:
    """Counts per status, latency and token statistics of a set of records."""
    recs = list(records)
    status = Counter(r.status for r in recs)
    lat = [r.latency_s for r in recs]
    comp = [r.usage.get("completion_tokens") or 0 for r in recs if r.usage]
    prompt = [r.usage.get("prompt_tokens") or 0 for r in recs if r.usage]
    reasoning = [r.reasoning_chars for r in recs]

    def q(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(p * len(xs)))], 2)

    return {
        "units": len(recs),
        "members": sum(len(r.members) for r in recs),
        "status": dict(status),
        "ok_fraction": round(status.get(STATUS_OK, 0) / len(recs), 4) if recs else None,
        "latency_s": {"p50": q(lat, 0.5), "p95": q(lat, 0.95), "max": q(lat, 1.0)},
        "completion_tokens": {
            "mean": round(statistics.mean(comp), 1) if comp else None,
            "p95": q(comp, 0.95),
            "total": sum(comp),
        },
        "prompt_tokens": {
            "mean": round(statistics.mean(prompt), 1) if prompt else None,
            "total": sum(prompt),
        },
        "reasoning_chars": {
            "mean": round(statistics.mean(reasoning), 1) if reasoning else None,
            "zero": sum(1 for x in reasoning if x == 0),
        },
        "models": sorted({r.model for r in recs}),
        "template_hashes": sorted({r.template_hash for r in recs}),
        "vllm_versions": sorted({str(r.vllm_version) for r in recs}),
    }
