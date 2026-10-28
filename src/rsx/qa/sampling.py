"""sampling_config: which candidates are drawn, and how often. See docs/qa/sampling.md.

Layers, later ones win: DEFAULTS < qa_templates/<set>/sampling.yaml < dataset overlay `sampling:`
< the dict given to the generator / to sample_qa_from_item.

A config is compiled once into a SamplingState (one weight per candidate, counts per unit and
qtype, allowed phrasings); sampling then only slices arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .builders import get_builder
from .facts import FLAG_BIT, FLAGS
from .templates import QTYPES, deep_merge

if TYPE_CHECKING:
    from .generator import QAGenerator


class Rarity(BaseModel):
    """Concept weight (n_max / n_units[concept]) ** alpha, at most max_ratio. n_units = number of
    reports where the concept is present; n_max = the most frequent concept (weight 1).
    scope present: only candidates whose concept is present in the item (rare positives);
    all: also the No answers about rare concepts."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(0.5, ge=0.0)
    max_ratio: float = Field(50.0, ge=1.0)
    scope: Literal["present", "all"] = "present"


class WhichPresent(BaseModel):
    """choice_which_present (builders with drawn options). A concept can be the answer only if it
    is negated in >= min_negated reports (so that it also appears as a distractor). With
    balance > 0, answer and distractor weights are both scaled by
    ((n_present + 1) / (n_negated + 1)) ** (-+balance), so that a concept is the answer and a
    distractor in similar proportions: the options alone do not give the answer away."""

    model_config = ConfigDict(extra="forbid")

    min_negated: int = Field(20, ge=0)
    balance: float = Field(1.0, ge=0.0)


class AnswerBalance(BaseModel):
    """Make the answers of each (template, concept) equally frequent (beta = 1; 0 = off).

    Start: weight (n_tc / (K * n_tca)) ** beta (K answers seen, n_tca candidates with that answer,
    from the reference stats). Then `iterations` rounds of calibration on the expected number of
    times each candidate is drawn (k questions per item, every item once): a weight only acts
    within its item, so frequent candidates must be pushed down for the rare ones to catch up.
    Factors are clipped to [1 / max_ratio, max_ratio].
    min_count > 0: a (template, concept) needs 2 answers with >= min_count candidates each,
    else it is never drawn (a concept that is always Yes teaches 'concept -> Yes')."""

    model_config = ConfigDict(extra="forbid")

    beta: float = Field(1.0, ge=0.0)
    max_ratio: float = Field(20.0, ge=1.0)
    min_count: int = Field(0, ge=0)
    iterations: int = Field(10, ge=0)
    k: int = Field(8, ge=1)  # questions per item assumed by the calibration
    # Weights only choose among an item's candidates: an item whose only noul candidates are No
    # still fills its noul quota with them. drop_excess keeps a drawn candidate with probability
    # (fewest expected draws among the answers of its (template, concept) / its own) ** beta:
    # balanced answers, at the price of items returning fewer than k questions.
    drop_excess: bool = False


class ItemWeights(BaseModel):
    """item_weights(): max | sum of the rarity weights of the concepts present in the item."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["max", "sum"] = "max"


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qtype: dict[str, float] = Field(
        default_factory=lambda: {"noul": 0.4, "choice": 0.3, "score": 0.3}
    )
    templates: dict[str, float] = Field(default_factory=dict)  # template_id -> multiplier
    rarity: Rarity = Field(default_factory=Rarity)
    answer_balance: AnswerBalance = Field(default_factory=AnswerBalance)
    flag_weights: dict[str, float] = Field(default_factory=dict)  # flag -> multiplier (default 1)
    low_evidence_threshold: float = 90.0
    phrasings: Literal["train", "heldout", "all"] = "train"
    shuffle_choice_options: bool = True
    which_present: WhichPresent = Field(default_factory=WhichPresent)
    fill: Literal["short", "rephrase"] = "short"
    item_weights: ItemWeights = Field(default_factory=ItemWeights)

    @field_validator("qtype")
    @classmethod
    def _qtype(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(QTYPES)
        if unknown:
            raise ValueError(f"unknown qtype(s) {sorted(unknown)}; known {list(QTYPES)}")
        if any(x < 0 for x in v.values()) or sum(v.values()) <= 0:
            raise ValueError("qtype probabilities must be >= 0 with a positive sum")
        return v

    @field_validator("flag_weights")
    @classmethod
    def _flags(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(FLAGS)
        if unknown:
            raise ValueError(f"unknown flag(s) {sorted(unknown)}; known {list(FLAGS)}")
        if any(x < 0 for x in v.values()):
            raise ValueError("flag weights must be >= 0")
        return v

    @field_validator("templates")
    @classmethod
    def _templates(cls, v: dict[str, float]) -> dict[str, float]:
        if any(x < 0 for x in v.values()):
            raise ValueError("template weights must be >= 0")
        return v

    def flag_weight(self, flag: str) -> float:
        return self.flag_weights.get(flag, 1.0)


def resolve_config(*layers: dict[str, Any] | SamplingConfig | None) -> SamplingConfig:
    merged: dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, SamplingConfig):
            layer = layer.model_dump(exclude_unset=True)
        merged = deep_merge(merged, layer)
    return SamplingConfig.model_validate(merged)


@dataclass
class SamplingState:
    """A sampling config compiled against the candidates of one generator."""

    config: SamplingConfig
    w: np.ndarray  # float64 weight per candidate (0 = never drawn)
    logw: np.ndarray  # log(w), -inf where w = 0 (Gumbel top-k keys)
    npos: np.ndarray  # int64 per (unit, qtype): candidates with w > 0
    p_qtype: np.ndarray  # float64 per qtype, sums to 1
    rarity: np.ndarray  # float64 per concept
    phrasings: list[np.ndarray]  # per template: allowed phrasing indices
    displays: list[list[str]]  # per concept: allowed display names
    distractor_w: np.ndarray  # float64 per negated-pool entry (0 = excluded, e.g. closed_world)
    accept: np.ndarray | None  # float64 per candidate: probability to keep it once drawn


def _lookup(series: pd.Series, index: pd.MultiIndex, fallback: np.ndarray) -> np.ndarray:
    out = series.reindex(index).to_numpy(dtype=float)
    return np.where(np.isnan(out), fallback, out)


def _segment_sum(x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    return cs[offsets[1:]] - cs[offsets[:-1]]


def _expected_draws(gen: QAGenerator, w: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    """Approximate expected number of draws of each candidate when every item gets k questions:
    m = k * p_qtype (renormalized over the item's qtypes, capped by its candidates) draws per
    (unit, qtype), candidate j drawn with probability 1 - (1 - w_j / W)^m, times the items of
    its unit."""
    c = gen.cand
    nq = len(QTYPES)
    n_units = len(gen.units)
    seg = c.unit.astype(np.int64) * nq + c.qtype
    total = _segment_sum(w, c.offsets)
    npos = _segment_sum(w > 0, c.offsets).reshape(n_units, nq)
    avail = (npos > 0) * p
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.nan_to_num(avail / avail.sum(axis=1, keepdims=True))
        m = np.minimum(k * share, npos).ravel()
        frac = np.where(total[seg] > 0, w / total[seg], 0.0)
    prob = 1.0 - (1.0 - np.clip(frac, 0.0, 1.0)) ** m[seg]
    items = np.bincount(gen.item_unit, minlength=n_units)
    return prob * items[c.unit]


def _answer_balance(
    gen: QAGenerator, cfg: SamplingConfig, w: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """See AnswerBalance. Groups = (template, concept, answer) of the balanced templates.
    Returns the weights and the acceptance probabilities (drop_excess) or None."""
    ab = cfg.answer_balance
    c = gen.cand
    tids = np.array(gen.template_ids, dtype=object)
    concepts = gen.concepts.to_numpy()
    n_c = len(concepts)
    balanced = np.array([get_builder(t.builder).balanced for t in gen.spec.templates])
    n_a = int(c.answer.max(initial=0)) + 2
    key = (c.template.astype(np.int64) * n_c + c.concept) * n_a + (c.answer + 1)
    uniq, inv = np.unique(key, return_inverse=True)
    ut, uc, ua = uniq // (n_c * n_a), (uniq // n_a) % n_c, uniq % n_a - 1
    on = balanced[ut]
    _, tc = np.unique(ut * n_c + uc, return_inverse=True)  # (template, concept) of each group

    # start: candidate counts of the reference stats (this generator's counts as fallback)
    own = np.bincount(inv).astype(float)
    ref = gen.stats.n_answers
    n_tca = _lookup(ref, pd.MultiIndex.from_arrays([tids[ut], concepts[uc], ua]), own)
    idx2 = pd.MultiIndex.from_arrays([tids[ut], concepts[uc]])
    by_tc = ref.groupby(level=[0, 1])
    n_tc = _lookup(by_tc.sum(), idx2, np.bincount(tc, weights=n_tca)[tc])
    k_tc = _lookup(by_tc.size(), idx2, np.bincount(tc)[tc].astype(float))
    lo, hi = 1 / ab.max_ratio, ab.max_ratio
    b = np.clip((n_tc / (k_tc * np.maximum(n_tca, 1.0))) ** ab.beta, lo, hi)
    b = np.where(on, b, 1.0)

    # calibration on the expected draws
    if ab.beta > 0:
        for _ in range(ab.iterations):
            drawn = np.bincount(inv, weights=_expected_draws(gen, w * b[inv], p, ab.k))
            n_drawn = np.maximum(np.bincount(tc, weights=drawn > 0), 1.0)
            target = (np.bincount(tc, weights=drawn) / n_drawn)[tc]
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(drawn > 0, (target / drawn) ** ab.beta, 1.0)
            b = np.where(on, np.clip(b * factor, lo, hi), 1.0)

    if ab.min_count > 0:
        good = n_tca >= ab.min_count
        b = np.where(on & (np.bincount(tc, weights=good)[tc] < 2), 0.0, b)
    accept = None
    if ab.drop_excess and ab.beta > 0:
        drawn = np.bincount(inv, weights=_expected_draws(gen, w * b[inv], p, ab.k))
        floor = np.full(tc.max(initial=-1) + 1, np.inf)
        np.minimum.at(floor, tc[drawn > 0], drawn[drawn > 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(drawn > 0, np.minimum(floor[tc] / drawn, 1.0) ** ab.beta, 1.0)
        accept = np.where(on, a, 1.0)[inv]
    return w * b[inv], accept


def compile_state(gen: QAGenerator, cfg: SamplingConfig) -> SamplingState:
    c = gen.cand
    templates = gen.spec.templates
    concepts = gen.concepts.to_numpy()
    n_c = len(concepts)
    unknown = set(cfg.templates) - set(gen.template_ids)
    if unknown:
        raise ValueError(f"sampling_config.templates: unknown template(s) {sorted(unknown)}")

    # phrasings / displays of the requested split
    phrasings = [
        np.array(gen.spec.phrasings[t.id].indices(cfg.phrasings), np.int64) for t in templates
    ]
    displays = []
    for name in concepts:
        ps = gen.spec.displays.get(name)
        if name.startswith("other:") or ps is None:
            displays.append(gen.concept_display(gen.concepts.get_loc(name)))
            continue
        idx = ps.indices(cfg.phrasings) or ps.indices("all")  # no held-out name: use them all
        displays.append([ps.texts[i] for i in idx])

    # template multipliers (0 when no phrasing of the split)
    t_mult = np.array(
        [cfg.templates.get(t.id, 1.0) * (len(phrasings[i]) > 0) for i, t in enumerate(templates)]
    )
    w = t_mult[c.template].astype(float)

    # rarity per concept, from the reference stats (train) with this generator's counts as fallback
    n_units = _lookup(gen.stats.n_units, pd.Index(concepts), gen.own_stats.n_units.to_numpy())
    n_max = max(float(gen.stats.n_units.max()), 1.0)
    rarity = np.minimum(
        (n_max / np.maximum(n_units, 1.0)) ** cfg.rarity.alpha, cfg.rarity.max_ratio
    )
    w *= np.where(c.present | (cfg.rarity.scope == "all"), rarity[c.concept], 1.0)

    # quality flags
    for name in FLAGS:
        fw = cfg.flag_weight(name)
        if fw == 1.0:
            continue
        if name == "low_evidence":
            mask = c.fuzzy < cfg.low_evidence_threshold
        else:
            mask = (c.flags & FLAG_BIT[name]) > 0
        w[mask] *= fw

    # templates with drawn options (choice_which_present): see WhichPresent
    wp = cfg.which_present
    distractor_ok = np.ones(len(gen.neg_concept), bool)
    if cfg.flag_weight("closed_world") == 0:
        distractor_ok = ~gen.neg_closed_world
    n_neg = np.bincount(gen.neg_concept[distractor_ok], minlength=n_c)
    ratio = np.clip((gen.facts.n_units + 1.0) / (n_neg + 1.0), 1e-3, 1e3) ** wp.balance
    distractor_w = np.where(distractor_ok, ratio[gen.neg_concept], 0.0)
    n_dis = np.add.reduceat(
        np.append(distractor_ok, False).astype(np.int64), gen.neg_offsets[:-1]
    ) * (np.diff(gen.neg_offsets) > 0)
    for i, t in enumerate(templates):
        if gen.options[i] is None:
            m = c.template == i
            w[m] /= ratio[c.concept[m]]
            w[m & (n_dis[c.unit] < (t.n_options or 4) - 1)] = 0.0
            w[m & (n_neg[c.concept] < wp.min_negated)] = 0.0

    p = np.array([cfg.qtype.get(q, 0.0) for q in QTYPES], float)
    p /= p.sum()
    w, accept = _answer_balance(gen, cfg, w, p)
    npos = _segment_sum(w > 0, c.offsets).astype(np.int64)
    with np.errstate(divide="ignore"):
        logw = np.log(w)
    return SamplingState(
        config=cfg,
        w=w,
        logw=logw,
        npos=npos,
        p_qtype=p,
        rarity=rarity,
        phrasings=phrasings,
        displays=displays,
        distractor_w=distractor_w,
        accept=accept,
    )
