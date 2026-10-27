"""Export a run directory to flat tables (parquet + csv) with evidence verification.

exports/
  reports.{parquet,csv}     one row per member (e.g. VolumeName): status + scalar fields flattened
                            as section.sub.field; lists as JSON strings
  <table>.{parquet,csv}     one row per list item x member for each config.export_tables entry
                            (e.g. findings, negations), with <field>_verbatim / _normalized / _fuzzy
                            columns for evidence fields
  failures.csv              units without a valid extraction (status, error, raw output)
  summary.json              statuses, latencies, tokens, evidence match rates

Typed nulls stay null (NA): "not mentioned" is never turned into False or "".
Optional template hook: postprocess.py defining `postprocess(table: str, df) -> df`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .evidence import verify
from .results import STATUS_OK, Record, latest_records
from .stats import summarize
from .template import Template, load_template

log = logging.getLogger(__name__)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dicts -> dotted keys; lists -> JSON strings; scalars and None kept."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, key))
            elif isinstance(v, list):
                out[key] = json.dumps(v, ensure_ascii=False)
            else:
                out[key] = v
    return out


def _evidence_columns(row: dict[str, Any], text: str, fields: list[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for col, value in list(row.items()):
        if col.split(".")[-1] in fields:
            for k, v in verify(value, text).items():
                extra[f"{col}_{k}"] = v
    return extra


def load_postprocess(template: Template):
    path = template.postprocess_path()
    if not path:
        return None
    spec = importlib.util.spec_from_file_location(f"rsx_postprocess_{template.name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, "postprocess", None)


def run_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.yaml"
    return (yaml.safe_load(path.read_text()) if path.exists() else None) or {}


def _guard(row: dict[str, Any], reserved: set[str], where: str) -> dict[str, Any]:
    """Template fields never overwrite bookkeeping columns: a clash is renamed <name>__field."""
    clash = reserved & set(row)
    if not clash:
        return row
    log.warning(
        "%s: template field(s) %s renamed to <name>__field (reserved column names)",
        where,
        sorted(clash),
    )
    return {(f"{k}__field" if k in clash else k): v for k, v in row.items()}


def build_tables(
    records: list[Record], template: Template, member_column: str, explode: bool = True
) -> dict[str, pd.DataFrame]:
    fields = template.config.evidence_fields
    tables: dict[str, list[dict[str, Any]]] = {"reports": []}
    for name in template.config.export_tables:
        tables[name] = []

    for rec in sorted(records, key=lambda r: r.members[0] if r.members else r.id):
        members = rec.members if explode else [";".join(rec.members)]
        parsed = rec.parsed or {}
        scalars = flatten(
            {k: v for k, v in parsed.items() if k not in template.config.export_tables.values()}
        )
        counts = {f"n_{k}": len(v) for k, v in parsed.items() if isinstance(v, list)}
        base = {
            "unit_id": rec.id,
            "status": rec.status,
            "attempts": rec.attempts,
            "error": rec.error,
            **counts,
        }
        scalars = _guard(scalars, {member_column, *base}, "reports")
        for m in members:
            tables["reports"].append({member_column: m, **base, **scalars})
        for name, key in template.config.export_tables.items():
            items = parsed.get(key) or []
            for idx, item in enumerate(items):
                row = flatten(item) if isinstance(item, dict) else {key: item}
                row.update(_evidence_columns(row, rec.text, fields))
                row = _guard(row, {member_column, "unit_id", "item_idx"}, name)
                for m in members:
                    tables[name].append(
                        {member_column: m, "unit_id": rec.id, "item_idx": idx, **row}
                    )

    dfs = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    # rows of one member (e.g. one VolumeName) together, items in extraction order
    for name, df in dfs.items():
        keys = [k for k in (member_column, "item_idx") if k in df.columns]
        if keys:
            dfs[name] = df.sort_values(keys, kind="stable").reset_index(drop=True)
    post = load_postprocess(template)
    if post:
        dfs = {name: post(name, df) for name, df in dfs.items()}
    return dfs


def evidence_rates(df: pd.DataFrame) -> dict[str, Any]:
    """Evidence match rates per extracted item (not weighted by the number of members)."""
    if {"unit_id", "item_idx"} <= set(df.columns):
        df = df.drop_duplicates(["unit_id", "item_idx"])
    out = {}
    for col in df.columns:
        if col.endswith("_normalized"):
            stem = col[: -len("_normalized")]
            n = len(df)
            out[stem] = {
                "rows": n,
                "verbatim": round(float(df[f"{stem}_verbatim"].mean()), 4) if n else None,
                "normalized": round(float(df[col].mean()), 4) if n else None,
                "fuzzy_ge_90": round(float((df[f"{stem}_fuzzy"] >= 90).mean()), 4) if n else None,
            }
    return out


def export_run(
    run_dir: Path, *, explode: bool = True, formats: tuple[str, ...] = ("parquet", "csv")
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    info = run_info(run_dir)
    template = load_template(run_dir / "template", name=info.get("template"))
    member_column = info.get("member_column", "id")
    records = list(latest_records(run_dir).values())
    if not records:
        raise RuntimeError(f"no records in {run_dir}")
    out_dir = run_dir / "exports"
    out_dir.mkdir(exist_ok=True)

    written: dict[str, Path] = {}
    dfs = build_tables(records, template, member_column, explode=explode)
    for name, df in dfs.items():
        for fmt in formats:
            path = out_dir / f"{name}.{fmt}"
            tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
            if fmt == "parquet":
                df.to_parquet(tmp, index=False)
            else:
                df.to_csv(tmp, index=False)
            os.replace(tmp, path)  # readers never see a half-written file
            written[f"{name}.{fmt}"] = path

    failures = [
        {
            "unit_id": r.id,
            "keys": ";".join(r.keys),
            "status": r.status,
            "error": r.error,
            "finish_reason": r.finish_reason,
            "raw": (r.raw or "")[:4000],
        }
        for r in records
        if r.status != STATUS_OK
    ]
    pd.DataFrame(
        failures, columns=["unit_id", "keys", "status", "error", "finish_reason", "raw"]
    ).to_csv(out_dir / ".failures.csv.tmp", index=False)
    os.replace(out_dir / ".failures.csv.tmp", out_dir / "failures.csv")
    written["failures.csv"] = out_dir / "failures.csv"

    summary = summarize(records)
    summary["tables"] = {name: len(df) for name, df in dfs.items()}
    summary["evidence"] = {
        name: evidence_rates(df) for name, df in dfs.items() if name != "reports"
    }
    summary["template"] = template.name
    summary["template_hash_now"] = template.hash
    (out_dir / ".summary.json.tmp").write_text(json.dumps(summary, indent=2))
    os.replace(out_dir / ".summary.json.tmp", out_dir / "summary.json")
    written["summary.json"] = out_dir / "summary.json"
    log.info("exported %s: %s", run_dir, {k: v for k, v in summary["tables"].items()})
    return written
