"""Exported tables of one structuring run (`rsx export`), read once for Q/A generation.

The exports repeat every row once per member (e.g. both CT-RATE reconstructions of a scan). The
store keeps one row per (unit_id, item_idx), plus the member -> unit_id table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_TABLES = ("reports", "findings", "negations")
OPTIONAL_TABLES = ("diagnoses", "limited_evaluation")
# JSON-encoded list columns (export.flatten), parsed once into tuples
LIST_COLUMNS = {
    "findings": ("lobes", "sites", "distribution", "descriptors"),
    "negations": ("regions", "sites"),
}


def read_export_table(exports_dir: str | Path, name: str) -> pd.DataFrame | None:
    """<name>.parquet, else <name>.csv, else None. Text columns become object columns: the
    pyarrow strings of pandas >= 3 make the many groupby / take of build_facts several times
    slower. No reader threads: they spin on busy login nodes and in DataLoader workers."""
    exports_dir = Path(exports_dir)
    parquet, csv = exports_dir / f"{name}.parquet", exports_dir / f"{name}.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet, use_threads=False)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return None
    text = [c for c in df.columns if df[c].dtype != object and pd.api.types.is_string_dtype(df[c])]
    return df.astype({c: object for c in text}) if text else df


def parse_list(value: Any) -> tuple:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return ()
    if isinstance(value, list | tuple | np.ndarray):
        return tuple(value)
    out = json.loads(value)
    return tuple(out) if isinstance(out, list) else ()


def as_bool(series: pd.Series) -> np.ndarray:
    """Boolean column that may hold NaN or strings (CSV round trip)."""
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"]).to_numpy()


class StructuredReportStore:
    """Exported tables at report-unit level.

    members  DataFrame [<member_column>, unit_id]: one row per member (e.g. per NIfTI volume)
    units    pd.Index of unit_ids (position = unit_idx)
    reports  one row per unit (exam.* columns)
    tables   {"findings"|"negations"|"diagnoses"|"limited_evaluation": DataFrame}, one row per
             (unit_id, item_idx), with an integer unit_idx column and list columns as tuples
    Only units with status ok are kept.
    """

    def __init__(self, tables: dict[str, pd.DataFrame | None], member_column: str = "VolumeName"):
        missing = [t for t in REQUIRED_TABLES if tables.get(t) is None]
        if missing:
            raise ValueError(f"missing export table(s) {missing}")
        reports = tables["reports"]
        if member_column not in reports.columns:
            raise ValueError(
                f"reports table has no member column {member_column!r}; "
                f"columns are {list(reports.columns)}"
            )
        if "status" in reports.columns:
            reports = reports[reports["status"] == "ok"]
        self.member_column = member_column
        self.members = (
            reports[[member_column, "unit_id"]]
            .drop_duplicates(member_column)
            .reset_index(drop=True)
        )
        self.units = pd.Index(sorted(self.members["unit_id"].unique()), name="unit_id")
        self.reports = (
            reports.drop(columns=[member_column])
            .drop_duplicates("unit_id")
            .set_index("unit_id")
            .reindex(self.units)
            .reset_index()
        )
        self.tables: dict[str, pd.DataFrame] = {}
        for name in REQUIRED_TABLES[1:] + OPTIONAL_TABLES:
            df = tables.get(name)
            if df is None:
                continue
            df = df[df["unit_id"].isin(self.units)].drop(columns=[member_column], errors="ignore")
            df = df.drop_duplicates(["unit_id", "item_idx"]).reset_index(drop=True)
            for col in LIST_COLUMNS.get(name, ()):
                if col in df.columns:
                    df[col] = df[col].map(parse_list)
            df.insert(0, "unit_idx", self.units.get_indexer(df["unit_id"]).astype(np.int32))
            self.tables[name] = df

    @classmethod
    def from_exports(
        cls, exports_dir: str | Path, member_column: str = "VolumeName"
    ) -> StructuredReportStore:
        exports_dir = Path(exports_dir)
        if not exports_dir.is_dir():
            raise FileNotFoundError(f"exports folder not found: {exports_dir}")
        tables = {
            name: read_export_table(exports_dir, name) for name in REQUIRED_TABLES + OPTIONAL_TABLES
        }
        store = cls(tables, member_column)
        store.source = str(exports_dir)
        return store

    source: str = ""

    @property
    def findings(self) -> pd.DataFrame:
        return self.tables["findings"]

    @property
    def negations(self) -> pd.DataFrame:
        return self.tables["negations"]

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in self.tables.items())
        return (
            f"StructuredReportStore({len(self.members)} members, {len(self.units)} units, {sizes})"
        )
