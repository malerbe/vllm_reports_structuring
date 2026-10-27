"""Export hook of chest_ct_findings (called by rsx export for each table).

findings: adds size_mm_max / size_mm_min (mm) computed from size.numbers + size.unit, and
          size_warning when the unit makes the value unusable (m, none).
"""

from __future__ import annotations

import json

import pandas as pd

UNIT_TO_MM = {"mm": 1.0, "cm": 10.0}


def _size_mm(numbers, unit):
    if numbers is None or (isinstance(numbers, float) and pd.isna(numbers)):
        return None, None, None
    values = json.loads(numbers) if isinstance(numbers, str) else list(numbers)
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None, None
    factor = UNIT_TO_MM.get(unit)
    if factor is None:
        return None, None, f"unusable unit {unit!r}"
    return max(values) * factor, min(values) * factor, None


def postprocess(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if table != "findings" or df.empty or "size.numbers" not in df.columns:
        return df
    sizes = [
        _size_mm(n, u)
        for n, u in zip(df["size.numbers"], df.get("size.unit", [None] * len(df)), strict=True)
    ]
    df = df.copy()
    df["size_mm_max"] = [s[0] for s in sizes]
    df["size_mm_min"] = [s[1] for s in sizes]
    df["size_warning"] = [s[2] for s in sizes]
    return df
