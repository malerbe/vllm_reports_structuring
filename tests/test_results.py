from __future__ import annotations

from pathlib import Path

import pytest

from rsx.data import WorkUnit
from rsx.results import (
    Record,
    ResultsWriter,
    RunLock,
    TemplateChangedError,
    latest_records,
    pending_units,
    results_filename,
)


def rec(uid: str, status: str = "ok", h: str = "h1", ts: str | None = None) -> Record:
    extra = {"ts": ts} if ts else {}
    return Record(
        id=uid, keys=[uid], members=[uid], text="t", status=status, template_hash=h, **extra
    )


def units(*ids: str) -> list[WorkUnit]:
    return [WorkUnit(id=i, text="t", members=(i,), keys=(i,)) for i in ids]


def test_writer_appends_and_repairs_torn_line(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    with ResultsWriter(path) as w:
        w.write(rec("a"))
    with path.open("ab") as f:
        f.write(b'{"id": "b", "trunc')  # job killed mid-write
    with ResultsWriter(path) as w:
        w.write(rec("c"))
    got = latest_records(tmp_path)
    assert set(got) == {"a", "c"}


def test_latest_record_wins(tmp_path: Path):
    with ResultsWriter(tmp_path / "results.jsonl") as w:
        w.write(rec("a", "invalid_json", ts="2026-01-01T00:00:00"))
        w.write(rec("a", "ok", ts="2026-01-02T00:00:00"))
    with ResultsWriter(tmp_path / results_filename(2, 1)) as w:
        w.write(rec("b", "ok", ts="2026-01-01T00:00:00"))
    got = latest_records(tmp_path)
    assert got["a"].status == "ok" and "b" in got


def test_pending_logic():
    records = {"a": rec("a"), "b": rec("b", "truncated")}
    assert [u.id for u in pending_units(units("a", "b", "c"), records, template_hash="h1")] == [
        "b",
        "c",
    ]
    assert [
        u.id
        for u in pending_units(
            units("a", "b", "c"), records, template_hash="h1", retry_failed=False
        )
    ] == ["c"]


def test_template_change_detected():
    records = {"a": rec("a", h="old")}
    with pytest.raises(TemplateChangedError):
        pending_units(units("a"), records, template_hash="new")
    assert [
        u.id
        for u in pending_units(units("a"), records, template_hash="new", allow_template_change=True)
    ] == ["a"]


def test_run_lock(tmp_path: Path):
    with RunLock(tmp_path / ".lock"), pytest.raises(RuntimeError, match="locked"):
        RunLock(tmp_path / ".lock").acquire()
    with RunLock(tmp_path / ".lock"):
        pass  # released


def test_results_filename():
    assert results_filename() == "results.jsonl"
    assert results_filename(4, 2) == "results.s2of4.jsonl"
