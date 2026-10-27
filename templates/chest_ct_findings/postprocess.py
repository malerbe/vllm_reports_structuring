"""Export hook of chest_ct_findings (called by rsx export for each table).

findings: adds size_mm_max / size_mm_min (mm) computed from size.numbers + size.unit, and
          size_warning when the unit makes the value unusable (m, none).
          Location checks (flags, nothing is corrected):
            lat_lobe_conflict   laterality contradicts the lobes (bilateral with lobes of one side,
                                right with a left lobe, ...)
            site_lobe_conflict  a lung segment outside the lobes / laterality of its entry
            site_unsupported    lung segments whose name is not in the sentence(s) of the evidence
            laterality_source   where the laterality is written: evidence, sentence, impression,
                                none (null laterality -> null)
            subset_of_invalid   subset_of does not point to an earlier entry of the same type
            location_alert      any of the above (laterality_source none counts)
"""

from __future__ import annotations

import json
import re

import pandas as pd
from rapidfuzz import fuzz

UNIT_TO_MM = {"mm": 1.0, "cm": 10.0}

LOBE_SIDE = {
    "right_upper_lobe": "right",
    "right_middle_lobe": "right",
    "right_lower_lobe": "right",
    "left_upper_lobe": "left",
    "lingula": "left",
    "left_lower_lobe": "left",
}
SEGMENT_LOBES = {
    "RUL": {"right_upper_lobe"},
    "RML": {"right_middle_lobe"},
    "RLL": {"right_lower_lobe"},
    "LUL": {"left_upper_lobe"},
    "LLL": {"left_lower_lobe"},
}
LINGULAR_LOBES = {"left_upper_lobe", "lingula"}
# segment name part -> words that must appear near the evidence
SEGMENT_WORDS = {
    "apical": r"\bapic(al|o)",
    "posterior": r"\bposter(ior|o)",
    "anterior": r"\banter(ior|o)",
    "lateral": r"\blater(al|o)",
    "medial": r"\bmedi(al|o)",
    "superior": r"\bsuperior\b",
    "lingular": r"lingul",
}
SIDE_WORDS = {"right": r"\bright\b", "left": r"\bleft\b|lingul", "both": r"\bboth\b|\bbilateral"}
_BOUNDARY = re.compile(r"(?<!\d)\.(?!\d)|\n")


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


def _list(value) -> list:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return value
    return []


def _null(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _context(evidence, text: str) -> str:
    """The sentence(s) of the report that contain the evidence (best fuzzy alignment)."""
    if not isinstance(evidence, str) or not evidence.strip() or not text:
        return ""
    lower = text.lower()
    start = lower.find(evidence.strip().lower())
    if start >= 0:
        end = start + len(evidence.strip())
    else:
        hit = fuzz.partial_ratio_alignment(evidence.lower(), lower)
        start, end = hit.dest_start, hit.dest_end
    before = [m.end() for m in _BOUNDARY.finditer(text, 0, start)]
    after = _BOUNDARY.search(text, end)
    return text[(before[-1] if before else 0) : (after.start() if after else len(text))]


def _impression(text: str) -> str:
    _, sep, rest = text.partition("IMPRESSION:")
    return rest if sep else ""


def _says(laterality: str, text: str) -> bool:
    found = {k for k, rx in SIDE_WORDS.items() if re.search(rx, text, re.I)}
    if laterality == "bilateral":
        return "both" in found or {"right", "left"} <= found
    return laterality in found


def laterality_source(laterality, evidence, sentence: str, impression: str) -> str | None:
    if _null(laterality):
        return None
    for source, text in (
        ("evidence", evidence),
        ("sentence", sentence),
        ("impression", impression),
    ):
        if isinstance(text, str) and _says(laterality, text):
            return source
    return "none"


def lat_lobe_conflict(laterality, lobes: list[str]) -> bool:
    sides = {LOBE_SIDE[lobe] for lobe in lobes if lobe in LOBE_SIDE}
    if laterality in ("right", "left"):
        return bool(sides - {laterality})
    if laterality == "bilateral":
        return len(sides) == 1
    return False


def _segment_lobes(site: str) -> set[str] | None:
    prefix, _, rest = site.partition("_")
    if prefix not in SEGMENT_LOBES:
        return None  # lymph node station or vessel
    return LINGULAR_LOBES if rest.endswith("lingular") else SEGMENT_LOBES[prefix]


def site_lobe_conflict(laterality, lobes: list[str], sites: list[str]) -> bool:
    for site in sites:
        allowed = _segment_lobes(site)
        if allowed is None:
            continue
        if lobes and not allowed & set(lobes):
            return True
        side = LOBE_SIDE[next(iter(allowed))]
        if laterality in ("right", "left") and side != laterality:
            return True
    return False


def site_unsupported(sites: list[str], context: str) -> str | None:
    missing = []
    for site in sites:
        if _segment_lobes(site) is None:
            continue
        parts = site.split("_", 1)[1].replace("apicoposterior", "apical").split("_")
        words = [SEGMENT_WORDS[p] for p in parts if p in SEGMENT_WORDS]
        if words and not any(re.search(w, context, re.I) for w in words):
            missing.append(site)
    return ",".join(missing) or None


def subset_of_invalid(df: pd.DataFrame) -> list[bool]:
    if "subset_of" not in df.columns:
        return [False] * len(df)
    keys = zip(df["unit_id"], df["item_idx"], strict=True)
    types = dict(zip(keys, df["finding_type"], strict=True))
    out = []
    for unit, idx, target, ftype in zip(
        df["unit_id"], df["item_idx"], df["subset_of"], df["finding_type"], strict=True
    ):
        if _null(target):
            out.append(False)
            continue
        target = int(target)
        out.append(not (0 <= target < idx and types.get((unit, target)) == ftype))
    return out


def _location_checks(df: pd.DataFrame, texts: dict[str, str]) -> pd.DataFrame:
    lat_conflict, site_conflict, unsupported, lat_source = [], [], [], []
    for row in df.to_dict("records"):
        text = texts.get(row.get("unit_id"), "")
        lobes, sites = _list(row.get("lobes")), _list(row.get("sites"))
        laterality, evidence = row.get("laterality"), row.get("evidence")
        context = _context(evidence, text)
        lat_conflict.append(lat_lobe_conflict(laterality, lobes))
        site_conflict.append(site_lobe_conflict(laterality, lobes, sites))
        unsupported.append(site_unsupported(sites, context))
        lat_source.append(laterality_source(laterality, evidence, context, _impression(text)))
    df["lat_lobe_conflict"] = lat_conflict
    df["site_lobe_conflict"] = site_conflict
    df["site_unsupported"] = unsupported
    df["laterality_source"] = lat_source
    df["subset_of_invalid"] = subset_of_invalid(df)
    df["location_alert"] = (
        df["lat_lobe_conflict"]
        | df["site_lobe_conflict"]
        | df["site_unsupported"].notna()
        | (df["laterality_source"] == "none")
        | df["subset_of_invalid"]
    )
    return df


def postprocess(table: str, df: pd.DataFrame, texts: dict[str, str] | None = None) -> pd.DataFrame:
    if table != "findings" or df.empty:
        return df
    df = df.copy()
    if "size.numbers" in df.columns:
        sizes = [
            _size_mm(n, u)
            for n, u in zip(df["size.numbers"], df.get("size.unit", [None] * len(df)), strict=True)
        ]
        df["size_mm_max"] = [s[0] for s in sizes]
        df["size_mm_min"] = [s[1] for s in sizes]
        df["size_warning"] = [s[2] for s in sizes]
    if "finding_type" in df.columns:
        df = _location_checks(df, texts or {})
    return df
