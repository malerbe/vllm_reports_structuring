"""Answer builders: turn the facts into the Q/A candidates of one template.

A builder returns one row per candidate question:
    unit_idx, concept_idx, slot, answer, flags, fuzzy, evidence, present
slot     builder-specific index filled into the phrasing (presence_side: the side), else -1
answer   index into the template's options; -1 when the options are drawn at sampling time
present  the concept is present in the unit (used for item weights)

Add a builder: write a function (facts, template) -> DataFrame and decorate it with
@register_builder(name, slots=..., options=...).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .facts import SIDES, Facts
from .templates import TemplateSpec, apply_where

COLUMNS = ["unit_idx", "concept_idx", "slot", "answer", "flags", "fuzzy", "evidence", "present"]


@dataclass(frozen=True)
class Builder:
    name: str
    build: Callable[[Facts, TemplateSpec], pd.DataFrame]
    slots: tuple[str, ...]  # placeholders the phrasings may use (besides {finding})
    # fixed options of a template (None: drawn per question at sampling time)
    options: Callable[[TemplateSpec, list[str]], list[str] | None]
    balanced: bool = True  # answers are fixed option indices: answer balancing applies


BUILDERS: dict[str, Builder] = {}


def _noul_options(tpl: TemplateSpec, noul_options: list[str]) -> list[str]:
    return list(noul_options)


def option_values(tpl: TemplateSpec) -> tuple[list, list[str]]:
    """answer.options: {value: label, ...} or [value, ...] (label = value with spaces)."""
    opts = tpl.answer.get("options")
    if not opts:
        raise ValueError(f"template {tpl.id!r}: answer.options is required")
    if isinstance(opts, dict):
        return list(opts), [str(v) for v in opts.values()]
    return list(opts), [str(v).replace("_", " ") for v in opts]


def register_builder(
    name: str,
    slots: tuple[str, ...] = (),
    options: Callable[[TemplateSpec, list[str]], list[str] | None] = _noul_options,
    balanced: bool = True,
):
    def deco(fn: Callable[[Facts, TemplateSpec], pd.DataFrame]):
        if name in BUILDERS:
            raise ValueError(f"builder {name!r} registered twice")
        BUILDERS[name] = Builder(name, fn, slots, options, balanced)
        return fn

    return deco


def get_builder(name: str) -> Builder:
    if name not in BUILDERS:
        raise KeyError(f"unknown builder {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name]


def candidate_rows(src: pd.DataFrame, answer, slot=-1, present=None) -> pd.DataFrame:
    out = src[["unit_idx", "concept_idx", "flags", "fuzzy", "evidence"]].copy()
    out["slot"] = slot
    out["answer"] = answer
    out["present"] = (out["answer"] == 1) if present is None else present
    return out[COLUMNS]


@register_builder("presence")
def presence(facts: Facts, tpl: TemplateSpec) -> pd.DataFrame:
    """Noul: is the concept present? Yes = a finding, No = a usable negation."""
    src = apply_where(facts.state, tpl.where, tpl.id)
    return candidate_rows(src, src["answer"].to_numpy())


@register_builder("presence_side", slots=("side",))
def presence_side(facts: Facts, tpl: TemplateSpec) -> pd.DataFrame:
    """Noul: is the concept present on {side}? answer.sides maps right/left to the slot text."""
    src = apply_where(facts.side_state, tpl.where, tpl.id)
    slot = src["side"].map({s: i for i, s in enumerate(SIDES)}).to_numpy()
    return candidate_rows(src, src["answer"].to_numpy(), slot=slot)


def _field_options(tpl: TemplateSpec, noul_options: list[str]) -> list[str]:
    return option_values(tpl)[1]


@register_builder("field", options=_field_options)
def field(facts: Facts, tpl: TemplateSpec) -> pd.DataFrame:
    """Choice / Score on an aggregated attribute (facts.agg column answer.field).
    answer.map renames values first (e.g. lingula -> left_upper_lobe); values not in the options
    give no candidate."""
    col = tpl.answer.get("field")
    if not col:
        raise ValueError(f"template {tpl.id!r}: answer.field is required")
    values, _ = option_values(tpl)
    src = apply_where(facts.agg, tpl.where, tpl.id)
    if col not in src.columns:
        raise ValueError(f"template {tpl.id!r}: unknown field {col!r}")
    v = src[col]
    if tpl.answer.get("map"):
        v = v.replace(tpl.answer["map"])
    pos = {val: i for i, val in enumerate(values)}
    answer = v.map(pos)
    keep = answer.notna()
    return candidate_rows(src[keep], answer[keep].astype(int).to_numpy(), present=True)


def _bin_options(tpl: TemplateSpec, noul_options: list[str]) -> list[str]:
    return [str(x) for x in tpl.answer["labels"]]


@register_builder("bins", options=_bin_options)
def bins(facts: Facts, tpl: TemplateSpec) -> pd.DataFrame:
    """Score on a number (answer.field) cut by answer.edges; bin i is edges[i-1] <= x < edges[i]."""
    col, edges, labels = (tpl.answer.get(k) for k in ("field", "edges", "labels"))
    if not col or not edges or not labels:
        raise ValueError(f"template {tpl.id!r}: answer needs field, edges and labels")
    if len(labels) != len(edges) + 1 or list(edges) != sorted(edges):
        raise ValueError(f"template {tpl.id!r}: needs sorted edges and len(edges) + 1 labels")
    src = apply_where(facts.agg, tpl.where, tpl.id)
    src = src[src[col].notna()]
    answer = np.searchsorted(np.asarray(edges, float), src[col].astype(float), side="right")
    return candidate_rows(src, answer, present=True)


def _dynamic_options(tpl: TemplateSpec, noul_options: list[str]) -> None:
    return None


@register_builder("present_among_negated", options=_dynamic_options, balanced=False)
def present_among_negated(facts: Facts, tpl: TemplateSpec) -> pd.DataFrame:
    """Choice: which of these findings is seen? One present concept + (n_options - 1) concepts
    negated in the same unit, drawn at sampling time. Units with too few negations give none."""
    n_options = tpl.n_options or 4
    present = facts.state[facts.state["answer"] == 1]
    src = apply_where(present, tpl.where, tpl.id)
    n_neg = facts.negated().groupby("unit_idx").size()
    src = src[src["unit_idx"].map(n_neg).fillna(0).to_numpy() >= n_options - 1]
    return candidate_rows(src, -1, present=True)
