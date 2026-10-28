"""Q/A template set: qa_templates/<name>/{concepts,templates,sampling}.yaml.

concepts.yaml   how findings become concepts (display names, `other` names, which negations count)
templates.yaml  question templates: type, builder, `where` filter, answer options, phrasings
sampling.yaml   default sampling_config
datasets/<dataset>.yaml  optional overlay for one dataset (loaded by the generator of that name)

Phrasings (and concept display names) are split into train / heldout, to measure how well a model
generalizes to wordings it never saw. The split of a text is a hash of (seed, key, text), so it does
not change when other phrasings are added; `split:` in the YAML forces it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..paths import REPO_ROOT

QA_TEMPLATES_DIR = REPO_ROOT / "qa_templates"
QTYPES = ("noul", "choice", "score")
SPLITS = ("train", "heldout")


class Phrasing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    split: Literal["train", "heldout"] | None = None  # None: decided by hash


def _phrasing_list(value: Any) -> Any:
    if isinstance(value, list):
        return [{"text": v} if isinstance(v, str) else v for v in value]
    return value


class Holdout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fraction: float = Field(0.0, ge=0.0, lt=1.0)
    seed: int = 0


class TemplateSpec(BaseModel):
    """One question template. `answer` holds the builder's parameters (field, options, edges...)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    qtype: Literal["noul", "choice", "score"]
    builder: str
    where: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    n_options: int | None = None
    phrasings: list[Phrasing]

    _phrasings = field_validator("phrasings", mode="before")(_phrasing_list)


class NegationRule(BaseModel):
    """Fix a systematic extraction error on negations, matched on the evidence (regex search,
    case-insensitive). action drop: the negation is ignored; retype: it negates `to` instead."""

    model_config = ConfigDict(extra="forbid")

    finding_type: str | list[str]
    evidence: str | None = None  # the rule applies if the evidence matches (None: always)
    evidence_not: str | None = None  # ... and does not match this
    action: Literal["drop", "retype"]
    to: str | None = None


class FindingRule(BaseModel):
    """Fix a systematic extraction error on present findings, matched on the evidence.
    action add: also add a present finding of type `to` (location kept, flag `derived`);
    negate: the finding becomes a negation of `to` (and is no longer a finding)."""

    model_config = ConfigDict(extra="forbid")

    finding_type: str | list[str]
    evidence: str | None = None
    evidence_not: str | None = None
    # only findings whose short axis (size_mm_min) is missing or <= this value
    max_short_axis_mm: float | None = None
    action: Literal["add", "negate"]
    to: str


class ClosedWorldCondition(BaseModel):
    """Conditional closed world: "not mentioned" counts as No only if the report has an entry
    (finding or negation, of any type) in one of `regions`, whose evidence matches `evidence`
    if given, and no `other` finding in these regions (an unnamed abnormality there)."""

    model_config = ConfigDict(extra="forbid")

    regions: list[str]
    evidence: str | None = None


class ConceptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # `other` findings become the concept "other:<normalized name>" if the name is found in at
    # least this many report units; rarer `other` findings are not used
    other_min_count: int = 30
    negation_kinds: list[str] = Field(
        default_factory=lambda: ["explicit_negation", "normal_statement"]
    )
    # qualifiers that do not narrow a negation (regex, full match, lower case), per finding_type;
    # `_all` applies to every type. Any other qualifier makes the negation unusable.
    neutral_qualifiers: dict[str, list[str]] = Field(default_factory=dict)
    # child finding_type -> parent finding_types it is a kind of: a present child blocks the
    # negations of its parents ('no infiltration' + ground-glass densities: lung_opacity unknown)
    implies: dict[str, list[str]] = Field(default_factory=dict)
    # finding_types for which "not mentioned" counts as No (closed world). Empty by default: a No
    # comes from an explicit negation only. These No carry the closed_world flag. A concept that
    # is uncertain, or negated only with a narrowing qualifier, stays unknown.
    closed_world: list[str] = Field(default_factory=list)
    # finding_type -> condition: closed world only in reports that mention the region
    closed_world_if: dict[str, ClosedWorldCondition] = Field(default_factory=dict)
    # systematic extraction errors, fixed before the concepts are built
    negation_rules: list[NegationRule] = Field(default_factory=list)
    finding_rules: list[FindingRule] = Field(default_factory=list)
    # concepts never asked about (names or glob patterns, e.g. "other:*")
    exclude: list[str] = Field(default_factory=list)
    display_holdout: Holdout = Field(default_factory=Holdout)
    # finding_type -> display names; one of the requested split is drawn per use
    display: dict[str, list[Phrasing]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _closed_world_lists(self) -> ConceptSpec:
        dupes = sorted({t for t in self.closed_world if self.closed_world.count(t) > 1})
        both = sorted(set(self.closed_world) & set(self.closed_world_if))
        if dupes or both:
            raise ValueError(
                f"closed_world: duplicated {dupes}; in both closed_world and closed_world_if {both}"
            )
        return self

    @field_validator("display", mode="before")
    @classmethod
    def _displays(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _phrasing_list(v if isinstance(v, list) else [v]) for k, v in value.items()}
        return value


class TemplatesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrasing_holdout: Holdout = Field(default_factory=Holdout)
    noul_options: list[str] = Field(default_factory=lambda: ["No", "Yes"])
    templates: list[TemplateSpec]


def text_hash(*parts: Any) -> str:
    return hashlib.sha1("\0".join(str(p) for p in parts).encode()).hexdigest()


def assign_splits(key: str, phrasings: list[Phrasing], holdout: Holdout) -> list[str]:
    """train / heldout per phrasing. At least one stays train; with fraction > 0 and 2+
    phrasings, at least one is held out (the one with the lowest hash)."""
    u = [int(text_hash(holdout.seed, key, p.text)[:12], 16) / 16**12 for p in phrasings]
    splits = [
        p.split or ("heldout" if ui < holdout.fraction else "train")
        for p, ui in zip(phrasings, u, strict=True)
    ]
    free = [i for i, p in enumerate(phrasings) if p.split is None]
    if phrasings and "train" not in splits:
        if not free:
            raise ValueError(f"{key}: every phrasing is forced to heldout; keep one in train")
        splits[max(free, key=lambda i: u[i])] = "train"
    if holdout.fraction > 0 and len(phrasings) > 1 and "heldout" not in splits:
        movable = [i for i in free if splits.count("train") > 1]
        if movable:
            splits[min(movable, key=lambda i: u[i])] = "heldout"
    return splits


@dataclass
class PhrasingSet:
    """Texts of one template (or one concept's display names) with their ids and splits."""

    key: str
    texts: list[str]
    ids: list[str]
    splits: list[str]

    @classmethod
    def build(cls, key: str, phrasings: list[Phrasing], holdout: Holdout) -> PhrasingSet:
        texts = [p.text for p in phrasings]
        return cls(
            key=key,
            texts=texts,
            ids=[text_hash(key, t)[:10] for t in texts],
            splits=assign_splits(key, phrasings, holdout),
        )

    def indices(self, split: str) -> list[int]:
        """split: train | heldout | all."""
        if split == "all":
            return list(range(len(self.texts)))
        if split not in SPLITS:
            raise ValueError(f"phrasings must be train, heldout or all, got {split!r}")
        return [i for i, s in enumerate(self.splits) if s == split]


@dataclass
class QASpec:
    """A loaded template set."""

    name: str
    path: Path
    concepts: ConceptSpec
    templates: list[TemplateSpec]
    noul_options: list[str]
    phrasings: dict[str, PhrasingSet]  # template id -> phrasings
    displays: dict[str, PhrasingSet]  # finding_type -> display names
    sampling: dict[str, Any] = field(default_factory=dict)
    overlay: Path | None = None

    def template(self, template_id: str) -> TemplateSpec:
        for t in self.templates:
            if t.id == template_id:
                return t
        raise KeyError(f"unknown template {template_id!r}")

    def phrasing_table(self) -> pd.DataFrame:
        """One row per phrasing (template_id, phrasing_id, split, text)."""
        rows = [
            {"template_id": key, "phrasing_id": pid, "split": s, "text": t}
            for key, ps in self.phrasings.items()
            for t, pid, s in zip(ps.texts, ps.ids, ps.splits, strict=True)
        ]
        return pd.DataFrame(rows)


def resolve_spec_dir(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.is_dir():
        return p
    d = QA_TEMPLATES_DIR / str(name_or_path)
    if d.is_dir():
        return d
    raise FileNotFoundError(
        f"Q/A template set {name_or_path!r} not found (folder, or a name under {QA_TEMPLATES_DIR})"
    )


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.exists() else None


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        out[k] = (
            deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


OVERLAY_KEYS = {"concepts", "templates", "sampling"}


def apply_overlay(
    concepts: dict[str, Any],
    templates: dict[str, Any],
    sampling: dict[str, Any],
    overlay: dict[str, Any],
    where: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Dataset overlay (qa_templates/<set>/datasets/<dataset>.yaml):
    concepts:  {key: value}              each key replaces the concepts.yaml key
    templates: {template_id: {key: v}}   each key replaces the template's key; null drops it
    sampling:  {...}                     merged into sampling.yaml
    """
    unknown = set(overlay) - OVERLAY_KEYS
    if unknown:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown)}; allowed {sorted(OVERLAY_KEYS)}"
        )
    concepts = {**concepts, **(overlay.get("concepts") or {})}
    by_id = {t["id"]: t for t in templates["templates"]}
    for tid, change in (overlay.get("templates") or {}).items():
        if tid not in by_id:
            raise ValueError(f"{where}: unknown template {tid!r}")
        if change is None:
            del by_id[tid]
        else:
            by_id[tid] = {**by_id[tid], **change}
    templates = {**templates, "templates": list(by_id.values())}
    return concepts, templates, deep_merge(sampling, overlay.get("sampling") or {})


def load_qa_spec(name_or_path: str | Path, overlay: str | Path | None = None) -> QASpec:
    """Load a template set; overlay: a dataset overlay YAML applied on top (see apply_overlay)."""
    d = resolve_spec_dir(name_or_path)
    tfile = d / "templates.yaml"
    if not tfile.exists():
        raise FileNotFoundError(f"{tfile} not found")
    raw_templates = _read_yaml(tfile) or {}
    raw_concepts = _read_yaml(d / "concepts.yaml") or {}
    sampling = _read_yaml(d / "sampling.yaml") or {}
    if overlay is not None:
        overlay = Path(overlay)
        if not overlay.exists():
            raise FileNotFoundError(f"overlay {overlay} not found")
        raw_concepts, raw_templates, sampling = apply_overlay(
            raw_concepts, raw_templates, sampling, _read_yaml(overlay) or {}, str(overlay)
        )
    tf = TemplatesFile.model_validate(raw_templates)
    ids = [t.id for t in tf.templates]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{tfile}: duplicate template id(s) {dupes}")
    for t in tf.templates:
        if not t.phrasings:
            raise ValueError(f"{tfile}: template {t.id!r} has no phrasings")
    concepts = ConceptSpec.model_validate(raw_concepts)
    return QASpec(
        name=d.name,
        path=d,
        concepts=concepts,
        templates=tf.templates,
        noul_options=tf.noul_options,
        phrasings={
            t.id: PhrasingSet.build(t.id, t.phrasings, tf.phrasing_holdout) for t in tf.templates
        },
        displays={
            k: PhrasingSet.build(f"display:{k}", v, concepts.display_holdout)
            for k, v in concepts.display.items()
        },
        sampling=sampling,
        overlay=overlay,
    )


def apply_where(df: pd.DataFrame, where: dict[str, Any], what: str = "") -> pd.DataFrame:
    """Rows of df matching every condition:
    col: notnull | null     value present / missing
    col: [a, b]             value in the list
    col: {not_in: [a, b]}   value not in the list (missing values pass)
    col: {min: x, max: y}   numeric bounds (inclusive)
    col: value              equality
    """
    mask = pd.Series(True, index=df.index)
    for col, cond in where.items():
        if col not in df.columns:
            raise ValueError(f"{what}: `where` column {col!r} not in {sorted(df.columns)}")
        s = df[col]
        if cond == "notnull":
            mask &= s.notna()
        elif cond == "null":
            mask &= s.isna()
        elif isinstance(cond, list):
            mask &= s.isin(cond)
        elif isinstance(cond, dict):
            unknown = set(cond) - {"not_in", "min", "max"}
            if unknown:
                raise ValueError(f"{what}: unknown `where` operator(s) {sorted(unknown)} for {col}")
            if "not_in" in cond:
                mask &= ~s.isin(cond["not_in"])
            if "min" in cond:
                mask &= s >= cond["min"]
            if "max" in cond:
                mask &= s <= cond["max"]
        else:
            mask &= s == cond
    return df[mask]
