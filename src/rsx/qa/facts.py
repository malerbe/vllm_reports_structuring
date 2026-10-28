"""Facts of each report unit: the material the Q/A builders turn into candidates.

concept     finding_type, or "other:<normalized name>" for frequent `other` findings
state       per (unit, concept): answer 1 = present (a finding with status present),
            0 = negated (a usable negation and no finding). Not mentioned = no row: unknown,
            except for the concepts of ConceptSpec.closed_world / closed_world_if (No, flag
            closed_world). Uncertain, or negated only with a narrowing qualifier = unknown.

ConceptSpec.negation_rules / finding_rules fix systematic extraction errors first.
side_state  per (unit, concept, side right/left): the same for one side
agg         per (unit, concept present): attributes aggregated over the unit's entries;
            an attribute is missing (NaN) when the entries do not determine it
"""

from __future__ import annotations

import fnmatch
import logging
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .store import StructuredReportStore, as_bool
from .templates import ConceptSpec

log = logging.getLogger(__name__)

# Quality flags carried from the exported entries to the Q/A candidates (bit i = FLAGS[i]).
# closed_world marks a No that comes from "not mentioned" (ConceptSpec.closed_world[_if]);
# derived a finding added by a ConceptSpec.finding_rules rule.
# low_evidence is not stored: it is derived at sampling time from the candidate's lowest
# evidence_fuzzy, so its threshold can change without recompiling.
ENTRY_FLAGS = ("location_alert", "lat_lobe_conflict", "site_lobe_conflict", "subset_of_invalid")
FLAGS = (*ENTRY_FLAGS, "size_warning", "closed_world", "derived", "low_evidence")
FLAG_BIT = {name: 1 << i for i, name in enumerate(FLAGS)}
STORED_FLAGS = FLAGS[:-1]
NOT_MENTIONED = "(not mentioned)"  # evidence of a closed-world No
SIDES = ("right", "left")
KEYS = ["unit_idx", "concept_idx"]

_ARTICLE = re.compile(r"^(a|an|the)\s+")
_SPACES = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lower case, no leading article, last word singular ('Accessory spleens' -> 'accessory
    spleen')."""
    s = _ARTICLE.sub("", _SPACES.sub(" ", str(name).lower()).strip(" .,;:"))
    words = s.split(" ")
    last = words[-1]
    if len(last) > 3 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
        last = last[:-1]
    words[-1] = last
    return " ".join(words)


def flag_names(bits: int) -> list[str]:
    return [name for name in FLAGS if bits & FLAG_BIT[name]]


def entry_flags(df: pd.DataFrame) -> np.ndarray:
    flags = np.zeros(len(df), np.uint8)
    for name in ENTRY_FLAGS:
        if name in df.columns:
            flags |= np.where(as_bool(df[name]), FLAG_BIT[name], 0).astype(np.uint8)
    if "size_warning" in df.columns:
        flags |= np.where(df["size_warning"].notna(), FLAG_BIT["size_warning"], 0).astype(np.uint8)
    if "derived" in df.columns:
        flags |= np.where(as_bool(df["derived"]), FLAG_BIT["derived"], 0).astype(np.uint8)
    return flags


def _types(rule) -> list[str]:
    return [rule.finding_type] if isinstance(rule.finding_type, str) else list(rule.finding_type)


def rule_mask(df: pd.DataFrame, rule) -> np.ndarray:
    """Rows of the rule's finding_type(s) whose evidence matches `evidence` (if given) and not
    `evidence_not` (if given); case-insensitive regex search."""
    m = df["finding_type"].isin(_types(rule)).to_numpy(copy=True)
    ev = df["evidence"].fillna("").astype(str)
    if rule.evidence:
        m = m & _search(ev, rule.evidence)
    if rule.evidence_not:
        m = m & ~_search(ev, rule.evidence_not)
    return m


def _search(text: pd.Series, pattern: str) -> np.ndarray:
    with warnings.catch_warnings():  # "has match groups": groups are fine in a search
        warnings.simplefilter("ignore", UserWarning)
        return text.str.contains(pattern, case=False, regex=True).to_numpy(bool)


# attributes that a finding added by a rule does not inherit (they describe the source finding)
_NOT_INHERITED = (
    "subset_of",
    "count",
    "count_number",
    "severity",
    "change",
    "size_mm_max",
    "size_mm_min",
    "size.measure",
    "size.qualifier",
    "size.numbers",
    "size.unit",
    "size.is_largest",
)


def apply_rules(
    f: pd.DataFrame, n: pd.DataFrame, spec: ConceptSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ConceptSpec.negation_rules then finding_rules (masks computed on the input tables)."""
    n = n.copy()
    for rule in spec.negation_rules:
        m = rule_mask(n, rule)
        if rule.action == "drop":
            n = n[~m]
        else:
            if not rule.to:
                raise ValueError(f"negation rule on {rule.finding_type}: retype needs `to`")
            n.loc[m, "finding_type"] = rule.to
    added, negated, drop = [], [], np.zeros(len(f), bool)
    current = ((f["status"] == "present") & (f["change"] != "resolved")).to_numpy()
    short = f["size_mm_min"] if "size_mm_min" in f.columns else pd.Series(np.nan, index=f.index)
    for rule in spec.finding_rules:
        m = rule_mask(f, rule) & current  # never from an uncertain or resolved finding
        if rule.max_short_axis_mm is not None:
            m &= ~(short > rule.max_short_axis_mm).to_numpy()
        if rule.action == "add":
            new = f[m].copy()
            new["finding_type"] = rule.to
            for col in _NOT_INHERITED:
                if col in new.columns:
                    new[col] = None
            new["derived"] = True
            added.append(new)
        else:
            src = f[m]
            negated.append(
                pd.DataFrame(
                    {
                        "unit_idx": src["unit_idx"].to_numpy(),
                        "unit_id": src["unit_id"].to_numpy(),
                        "evidence": src["evidence"].to_numpy(),
                        "finding_type": rule.to,
                        "kind": "normal_statement",
                        "qualifier": None,
                        "regions": [(r,) if isinstance(r, str) else () for r in src["region"]],
                        "laterality": src["laterality"].to_numpy(),
                        "sites": [()] * len(src),
                        "evidence_fuzzy": src["evidence_fuzzy"].to_numpy(),
                    }
                )
            )
            drop |= m
    f = pd.concat([f[~drop], *added], ignore_index=True) if added or drop.any() else f
    n = pd.concat([n, *negated], ignore_index=True) if negated else n
    return f, n


def _excluded(names: pd.Series, patterns: list[str]) -> np.ndarray:
    if not patterns:
        return np.zeros(len(names), bool)
    return names.map(lambda c: any(fnmatch.fnmatchcase(c, p) for p in patterns)).to_numpy(bool)


def neutral_qualifier_mask(neg: pd.DataFrame, spec: ConceptSpec) -> np.ndarray:
    """True where the negation has no qualifier, or one listed as neutral for its finding_type."""
    q = neg["qualifier"].astype("string").str.lower().str.strip()
    ok = q.isna().to_numpy().copy()
    common = spec.neutral_qualifiers.get("_all", [])
    for ftype in neg["finding_type"].unique():
        patterns = common + spec.neutral_qualifiers.get(ftype, [])
        if not patterns:
            continue
        rx = re.compile("|".join(f"(?:{p})" for p in patterns))
        rows = (neg["finding_type"] == ftype).to_numpy() & ~ok
        ok[rows] = [bool(rx.fullmatch(v)) for v in q[rows]]
    return ok


def group_flags(df: pd.DataFrame) -> pd.Series:
    """Bitwise OR of `flags` per (unit_idx, concept_idx)."""
    out = None
    for name in STORED_FLAGS:
        bit = FLAG_BIT[name]
        has = ((df["flags"] & bit) > 0).groupby([df[k] for k in KEYS]).max().astype(np.uint8) * bit
        out = has if out is None else out | has
    return out.astype(np.uint8)


def join_evidence(df: pd.DataFrame, keys: list[str] = KEYS) -> pd.Series:
    """Distinct evidences of each group, joined with ' | ' (Python only for groups of 2+)."""
    g = df.groupby(keys)["evidence"]
    out, size = g.first(), g.size()
    multi = size.index[size > 1]
    if len(multi):
        sub = df[pd.MultiIndex.from_frame(df[keys]).isin(multi)]
        joined = sub.groupby(keys)["evidence"].agg(lambda s: " | ".join(dict.fromkeys(s)))
        out.loc[joined.index] = joined
    return out


def _summary(df: pd.DataFrame, answer: int, flags: bool) -> pd.DataFrame:
    g = df.groupby(KEYS)
    out = g.agg(finding_type=("finding_type", "first"), fuzzy=("fuzzy", "min"))
    out["flags"] = group_flags(df) if flags else np.uint8(0)
    out["evidence"] = join_evidence(df)
    out["answer"] = answer
    return out


def _unique(df: pd.DataFrame, col: str) -> pd.Series:
    """The value of col if the non-missing values of the group agree, else NaN."""
    g = df.groupby(KEYS)[col]
    return g.first().where(g.nunique() == 1)


def _implied(
    pairs: pd.MultiIndex, concepts: pd.Index, implies: dict[str, list[str]]
) -> pd.MultiIndex:
    """(unit_idx, parent concept_idx) for every child (unit_idx, concept_idx) pair."""
    units, parents = [], []
    child_units = np.asarray(pairs.get_level_values("unit_idx"))
    child_concepts = np.asarray(pairs.get_level_values("concept_idx"))
    for child, targets in implies.items():
        c = concepts.get_indexer([child])[0]
        if c < 0:
            continue
        u = child_units[child_concepts == c]
        for p in concepts.get_indexer(targets):
            if p >= 0:
                units.append(u)
                parents.append(np.full(len(u), p, dtype=np.int32))
    if not units:
        return pd.MultiIndex.from_arrays([[], []], names=KEYS)
    return pd.MultiIndex.from_arrays([np.concatenate(units), np.concatenate(parents)], names=KEYS)


def _closed_world_rows(index: pd.MultiIndex, concepts: pd.Index) -> pd.DataFrame:
    """No answers for (unit, concept) pairs that are not mentioned."""
    out = pd.DataFrame(index=index)
    out["finding_type"] = concepts.to_numpy()[index.get_level_values("concept_idx")]
    out["fuzzy"] = np.float32(100.0)
    out["flags"] = np.uint8(FLAG_BIT["closed_world"])
    out["evidence"] = NOT_MENTIONED
    out["answer"] = 0
    return out


@dataclass
class Facts:
    concepts: pd.Index  # concept names (position = concept_idx)
    concept_type: np.ndarray  # finding_type of each concept ("other" for other:...)
    n_units: np.ndarray  # number of units where the concept is present
    state: pd.DataFrame
    side_state: pd.DataFrame
    agg: pd.DataFrame
    n_conflicts: int  # (unit, concept) both present and negated: counted as present

    def negated(self) -> pd.DataFrame:
        return self.state[self.state["answer"] == 0][KEYS]


def _pairs(df: pd.DataFrame, concepts: pd.Index) -> pd.MultiIndex:
    """(unit_idx, concept_idx) of the rows whose finding_type is a concept."""
    c = concepts.get_indexer(df["finding_type"])
    keep = c >= 0
    idx = pd.MultiIndex.from_arrays(
        [df["unit_idx"].to_numpy()[keep], c[keep].astype(np.int32)], names=KEYS
    )
    return idx.unique()


def _pairs_of(units_by_concept: dict[int, np.ndarray]) -> pd.MultiIndex:
    """(unit_idx, concept_idx) pairs from {concept_idx: unit indices}."""
    if not units_by_concept:
        return pd.MultiIndex.from_arrays([[], []], names=KEYS)
    u = np.concatenate([np.asarray(v, np.int64) for v in units_by_concept.values()])
    c = np.concatenate([np.full(len(v), k, np.int32) for k, v in units_by_concept.items()])
    return pd.MultiIndex.from_arrays([u, c], names=KEYS)


def _region_entries(f_all: pd.DataFrame, n_all: pd.DataFrame) -> pd.DataFrame:
    """[unit_idx, region, evidence, is_other] of every finding (any status) and negation."""
    fr = f_all[["unit_idx", "region", "evidence"]].assign(is_other=f_all["finding_type"] == "other")
    nr = n_all[["unit_idx", "regions", "evidence"]].explode("regions")
    nr = nr.rename(columns={"regions": "region"}).assign(is_other=False)
    return pd.concat([fr, nr], ignore_index=True).dropna(subset=["region"])


def _closed_world_units(
    spec: ConceptSpec, ftype: str, n_units: int, regions: pd.DataFrame
) -> np.ndarray:
    """Units where "not mentioned" counts as No for ftype (closed_world / closed_world_if)."""
    if ftype in spec.closed_world:
        return np.arange(n_units)
    cond = spec.closed_world_if[ftype]
    rows = regions[regions["region"].isin(cond.regions)]
    blocked = set(rows.loc[rows["is_other"], "unit_idx"])  # unnamed abnormality in the region
    if cond.evidence:
        rows = rows[_search(rows["evidence"].astype(str), cond.evidence)]
    return np.array(sorted(set(rows["unit_idx"]) - blocked), dtype=np.int64)


def build_facts(store: StructuredReportStore, spec: ConceptSpec) -> Facts:
    f_all, n_all = apply_rules(store.findings, store.negations, spec)

    # ---- present findings -> concepts
    f = f_all[(f_all["status"] == "present") & (f_all["change"] != "resolved")].copy()
    is_other = (f["finding_type"] == "other").to_numpy()
    f["concept"] = f["finding_type"]
    f.loc[is_other, "concept"] = "other:" + f.loc[is_other, "name"].map(normalize_name)
    other_units = f[is_other].groupby("concept")["unit_idx"].nunique()
    f = f[~f["concept"].isin(other_units.index[other_units < spec.other_min_count])]
    f = f[~_excluded(f["concept"], spec.exclude)]
    f["flags"] = entry_flags(f)
    f["fuzzy"] = f["evidence_fuzzy"].fillna(0.0).astype(np.float32)

    # ---- negations (no `other`: a negation has no name to make a concept from)
    n = n_all[n_all["kind"].isin(spec.negation_kinds) & (n_all["finding_type"] != "other")]
    n = n[~_excluded(n["finding_type"], spec.exclude)]
    neutral = neutral_qualifier_mask(n, spec)
    narrowed = n[~neutral]  # 'no SUSPICIOUS nodule': says nothing about the finding itself
    n = n[neutral].copy()
    n["concept"] = n["finding_type"]
    n["fuzzy"] = n["evidence_fuzzy"].fillna(0.0).astype(np.float32)

    listed = pd.Series([*spec.closed_world, *spec.closed_world_if], dtype=object)
    listed = set(listed[~_excluded(listed, spec.exclude)])  # may be all No in this export
    concepts = pd.Index(sorted(set(f["concept"]) | set(n["concept"]) | listed), name="concept")
    f["concept_idx"] = concepts.get_indexer(f["concept"]).astype(np.int32)
    n["concept_idx"] = concepts.get_indexer(n["concept"]).astype(np.int32)
    lat = n["laterality"]
    generic = n[lat.isna() | (lat == "bilateral")]
    lateral = n[lat.isin(SIDES)]

    # ---- state: present wins over a negation of the same concept or of a parent concept;
    # an uncertain finding ('cannot be excluded') makes the concept (and its parents) unknown
    pres = _summary(f, 1, flags=True)
    uncertain_any = _pairs(f_all[f_all["status"] == "uncertain"], concepts)
    uncertain_any = uncertain_any.append(_implied(uncertain_any, concepts, spec.implies))
    uncertain = uncertain_any.difference(pres.index)
    implied = _implied(pres.index, concepts, spec.implies)
    blocked = pres.index.append(implied).append(uncertain)
    neg = _summary(generic, 0, flags=False) if len(generic) else pres.iloc[:0]
    conflict = neg.index.isin(blocked)
    neg = neg[~conflict]
    lateral = lateral[~pd.MultiIndex.from_frame(lateral[KEYS]).isin(blocked.difference(pres.index))]
    # closed world: listed concepts neither present, implied, uncertain, negated nor negated
    # with a narrowing qualifier count as No
    cw_types = [t for t in [*spec.closed_world, *spec.closed_world_if] if t in concepts]
    unused = [t for t in cw_types if t not in set(f["concept"]) | set(n["concept"])]
    if unused:
        log.warning("closed world types never found in these exports (typo?): %s", unused)
    cw_units: dict[int, np.ndarray] = {}
    unknown = blocked.append(_pairs(narrowed, concepts)).append(neg.index)
    if cw_types:
        regions = _region_entries(f_all, n_all)
        for t in cw_types:
            cw_units[int(concepts.get_loc(t))] = _closed_world_units(
                spec, t, len(store.units), regions
            )
        grid = _pairs_of(cw_units)
        free = grid[~grid.isin(unknown)]
        rows = _closed_world_rows(free, concepts)
        # a lateral negation of the concept ('no pneumothorax on the right'): keep its evidence
        if len(lateral):
            lat_ev = join_evidence(lateral)
            known = rows.index.isin(lat_ev.index)
            rows.loc[known, "evidence"] = lat_ev.reindex(rows.index[known]) + " | " + NOT_MENTIONED
        neg = pd.concat([neg, rows])
    state = pd.concat([pres, neg]).sort_index().reset_index()

    # ---- side_state
    # an entry without laterality, or an uncertain one, leaves the side undetermined, for the
    # concept and its parents
    no_side = f[f["laterality"].isna()].groupby(KEYS).size().index
    unknown_side = no_side.append(_implied(no_side, concepts, spec.implies)).append(uncertain_any)
    # closed world, present on the other side only: in the closed-world units of the concept
    present_cw = pres.index.intersection(_pairs_of(cw_units))
    nlat = narrowed["laterality"]
    sides = []
    for side in SIDES:
        hit = f[f["laterality"].isin([side, "bilateral"])]
        yes = _summary(hit, 1, flags=True) if len(hit) else pres.iloc[:0]
        # not answerable on this side: a child present there, or a narrowed negation
        side_unknown = unknown_side.append(_implied(yes.index, concepts, spec.implies)).append(
            _pairs(narrowed[nlat.isna() | nlat.isin(["bilateral", side])], concepts)
        )
        lat_neg = lateral[lateral["laterality"] == side]
        lat_neg = _summary(lat_neg, 0, flags=False) if len(lat_neg) else neg.iloc[:0]
        lat_neg = lat_neg[~lat_neg.index.isin(yes.index) & ~lat_neg.index.isin(side_unknown)]
        side_neg = neg[~neg.index.isin(lat_neg.index) & ~neg.index.isin(yes.index)]
        no = [lat_neg, side_neg]
        if len(present_cw):
            other_side = present_cw[
                ~present_cw.isin(yes.index)
                & ~present_cw.isin(side_unknown)
                & ~present_cw.isin(lat_neg.index)
            ]
            no.append(_closed_world_rows(other_side, concepts))
        sides.append(pd.concat([yes, *no]).assign(side=side))
    side_state = pd.concat(sides).sort_index().reset_index()

    # ---- agg: attributes of present concepts
    agg = pres[["finding_type", "flags", "fuzzy", "evidence"]].copy()
    agg["n_entries"] = f.groupby(KEYS).size()
    lt = f["laterality"]
    t = (
        f[KEYS]
        .assign(null=lt.isna(), r=lt.eq("right"), l=lt.eq("left"), b=lt.eq("bilateral"))
        .groupby(KEYS)
        .max()
    )
    side = pd.Series(None, index=t.index, dtype=object)
    side[t["l"]] = "left"
    side[t["r"]] = "right"
    side[t["b"] | (t["r"] & t["l"])] = "bilateral"
    side[t["null"]] = None  # an entry without laterality: the side is not determined
    agg["laterality"] = side

    ex = f[[*KEYS, "lobes"]].explode("lobes")
    t = ex.assign(nolobe=ex["lobes"].isna()).groupby(KEYS)
    t = t.agg(nolobe=("nolobe", "max"), nlobes=("lobes", "nunique"), lobe=("lobes", "first"))
    agg["lobe"] = t["lobe"].where(~t["nolobe"] & (t["nlobes"] == 1))

    t = f[[*KEYS, "region"]].assign(null=f["region"].isna()).groupby(KEYS)
    t = t.agg(null=("null", "max"), n=("region", "nunique"), region=("region", "first"))
    agg["region"] = t["region"].where(~t["null"] & (t["n"] == 1))

    agg["severity"] = _unique(f, "severity")
    agg["change"] = _unique(f, "change")

    qualifier = f.get("size.qualifier", pd.Series(index=f.index, dtype=object))
    sized = f[f["size_mm_max"].notna() & ~qualifier.isin(["less_than", "greater_than"])]
    largest = sized.loc[sized.groupby(KEYS)["size_mm_max"].idxmax()].set_index(KEYS)
    agg["size_mm_max"] = largest["size_mm_max"]
    agg["size_measure"] = largest.get("size.measure")

    top = f[f["subset_of"].isna()]
    n_top = top.groupby(KEYS).size()
    one = top.set_index(KEYS).loc[n_top.index[n_top == 1]]
    count = one["count"].copy()
    count[one["count_number"] == 1] = "single"
    count[one["count_number"] >= 2] = "multiple"
    agg["count"] = count  # only when the unit has a single top-level entry of the concept
    agg = agg.reset_index()

    n_units = np.bincount(pres.index.get_level_values("concept_idx"), minlength=len(concepts))
    concept_type = np.array(
        ["other" if c.startswith("other:") else c for c in concepts], dtype=object
    )
    return Facts(
        concepts=concepts,
        concept_type=concept_type,
        n_units=n_units.astype(np.int64),
        state=state,
        side_state=side_state,
        agg=agg,
        n_conflicts=int(conflict.sum()),
    )
