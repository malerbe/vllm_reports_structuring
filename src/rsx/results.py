"""Append-only JSONL results, resume logic and run-directory lock."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .data import WorkUnit

log = logging.getLogger(__name__)

STATUS_OK = "ok"
# Model-side failures: recorded (with the raw output) and retried on the next run by default.
MODEL_FAILURES = ("invalid_json", "validation_error", "truncated", "empty", "request_error")


class Record(BaseModel):
    """One line of results*.jsonl = the outcome of one work unit."""

    id: str
    keys: list[str]
    members: list[str]
    text: str  # report text sent to the LLM (export checks evidence quotes against it)
    status: str
    attempts: int = 1
    parsed: dict[str, Any] | None = None
    raw: str | None = None
    finish_reason: str | None = None
    reasoning_chars: int = 0
    reasoning: str | None = None  # only with --keep-reasoning
    usage: dict[str, Any] | None = None
    latency_s: float = 0.0
    error: str | None = None
    model: str = ""
    template: str = ""
    template_hash: str = ""
    vllm_version: str | None = None
    job_id: str | None = None
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))


def results_filename(num_shards: int = 1, shard_id: int = 0) -> str:
    return "results.jsonl" if num_shards <= 1 else f"results.s{shard_id}of{num_shards}.jsonl"


class ResultsWriter:
    """Single writer (event-loop thread). Each record is one flushed line; fsync at most every
    fsync_every_s seconds and on close. A torn last line (killed job) is skipped when reading and
    the next append starts on a fresh line."""

    def __init__(self, path: Path, fsync_every_s: float = 10.0):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a+b")
        self._f.seek(0, os.SEEK_END)
        if self._f.tell() > 0:
            self._f.seek(-1, os.SEEK_END)
            if self._f.read(1) != b"\n":
                self._f.write(b"\n")
        self._fsync_every = fsync_every_s
        self._last_sync = time.monotonic()
        self.count = 0

    def write(self, rec: Record) -> None:
        self._f.write(rec.model_dump_json().encode() + b"\n")
        self._f.flush()
        self.count += 1
        now = time.monotonic()
        if now - self._last_sync >= self._fsync_every:
            os.fsync(self._f.fileno())
            self._last_sync = now

    def close(self) -> None:
        if not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()

    def __enter__(self) -> ResultsWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_records(run_dir: Path) -> Iterator[Record]:
    """All records of all results*.jsonl files; unparsable lines are skipped (and logged)."""
    for path in sorted(run_dir.glob("results*.jsonl")):
        bad = 0
        with path.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Record.model_validate(json.loads(line))
                except (ValueError, TypeError):
                    bad += 1
        if bad:
            log.warning("%s: %d unreadable line(s) skipped (interrupted write)", path.name, bad)


def latest_records(run_dir: Path) -> dict[str, Record]:
    """Latest record per unit id (by timestamp), across shard files."""
    out: dict[str, Record] = {}
    for rec in iter_records(run_dir):
        prev = out.get(rec.id)
        if prev is None or rec.ts >= prev.ts:
            out[rec.id] = rec
    return out


class TemplateChangedError(RuntimeError):
    pass


def pending_units(
    units: list[WorkUnit],
    records: dict[str, Record],
    *,
    template_hash: str,
    retry_failed: bool = True,
    allow_template_change: bool = False,
) -> list[WorkUnit]:
    """Units still to process.

    Done = latest record ok (or any model failure, with retry_failed=False).

    Raises TemplateChangedError when the run already holds records made with another template
    version, unless allow_template_change (then those units are redone).
    """
    other = {r.template_hash for r in records.values() if r.template_hash != template_hash}
    if other and not allow_template_change:
        raise TemplateChangedError(
            f"this run already has results made with another template version ({sorted(other)} != "
            f"{template_hash}). Use a new --run-name, or --allow-template-change to redo them."
        )
    out = []
    for u in units:
        r = records.get(u.id)
        changed = r is not None and r.template_hash != template_hash
        if r is None or changed or (r.status != STATUS_OK and retry_failed):
            out.append(u)
    return out


class MetaLock:
    """Short blocking lock shared by all shards of a run (snapshot, run.yaml, final export)."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> MetaLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class RunLock:
    """Exclusive non-blocking lock so two jobs never write the same results file."""

    def __init__(self, path: Path):
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            holder = self.path.read_text().strip() if self.path.exists() else "?"
            raise RuntimeError(f"{self.path} is locked by another process ({holder})") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID', '-')}\n".encode())
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> RunLock:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
