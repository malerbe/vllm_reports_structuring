"""QAGenerator: compiles every possible Q/A of a structured-report export once, then samples.

    gen = CTRATEGenerator("chest_ct_findings")
    gen.connect_structured_report("runs/.../ctrate_train_v31/exports")
    gen.sample_qa_from_item("train_10001_a_1.nii.gz", 8)

connect_structured_report reads the exports, builds the facts and runs every template's builder.
The result (the candidates) is kept in flat numpy arrays sorted by (unit, qtype), with offsets per
(unit, qtype): sampling only slices them. No Python object per candidate, so DataLoader workers
share the arrays copy-on-write.

Subclass per dataset (see datasets/ct_rate.py): member_column, item_table, normalize_query.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .builders import get_builder
from .facts import FLAG_BIT, FLAGS, SIDES, build_facts, flag_names
from .sampling import SamplingConfig, SamplingState, compile_state, resolve_config
from .store import StructuredReportStore
from .templates import QTYPES, QASpec, load_qa_spec, resolve_spec_dir

NQ = len(QTYPES)
_CW = FLAG_BIT["closed_world"]
_LOW = FLAG_BIT["low_evidence"]


@dataclass
class QA:
    """One question/answer. answer = index into options; for score the options are ordered
    (ordinal), for choice they are shuffled (unless shuffle_choice_options is false)."""

    question: str
    qtype: str  # noul | choice | score
    options: list[str]
    answer: int
    item: str
    template_id: str
    phrasing_id: str
    concept: str
    flags: list[str]
    evidence: str

    @property
    def ordinal(self) -> bool:
        return self.qtype == "score"

    @property
    def answer_text(self) -> str:
        return self.options[self.answer]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ordinal": self.ordinal}


@dataclass
class Candidates:
    """All candidate questions, sorted by (unit, qtype, template, concept, slot).

    The candidates of unit u and qtype q are [offsets[u*NQ+q], offsets[u*NQ+q+1]).
    """

    unit: np.ndarray  # int32
    template: np.ndarray  # int16, index into spec.templates
    qtype: np.ndarray  # int8, index into QTYPES
    concept: np.ndarray  # int32, index into concepts
    slot: np.ndarray  # int8
    answer: np.ndarray  # int16, -1 = options drawn at sampling time
    flags: np.ndarray  # uint8, FLAGS bits (low_evidence excluded)
    fuzzy: np.ndarray  # float32, lowest evidence_fuzzy of the entries used
    present: np.ndarray  # bool
    offsets: np.ndarray  # int64, n_units * NQ + 1
    ev_offsets: np.ndarray  # int64, len + 1
    ev_bytes: np.ndarray  # uint8, UTF-8 evidence texts back to back

    @classmethod
    def from_frame(cls, df: pd.DataFrame, n_units: int) -> Candidates:
        df = df.sort_values(["unit_idx", "qtype", "template", "concept_idx", "slot"], kind="stable")
        unit = df["unit_idx"].to_numpy(np.int32)
        qtype = df["qtype"].to_numpy(np.int8)
        key = unit.astype(np.int64) * NQ + qtype
        encoded = [s.encode() for s in df["evidence"]]
        ev_offsets = np.zeros(len(encoded) + 1, np.int64)
        np.cumsum([len(b) for b in encoded], out=ev_offsets[1:])
        return cls(
            unit=unit,
            template=df["template"].to_numpy(np.int16),
            qtype=qtype,
            concept=df["concept_idx"].to_numpy(np.int32),
            slot=df["slot"].to_numpy(np.int8),
            answer=df["answer"].to_numpy(np.int16),
            flags=df["flags"].to_numpy(np.uint8),
            fuzzy=df["fuzzy"].to_numpy(np.float32),
            present=df["present"].to_numpy(bool),
            offsets=np.searchsorted(key, np.arange(n_units * NQ + 1)).astype(np.int64),
            ev_offsets=ev_offsets,
            ev_bytes=np.frombuffer(b"".join(encoded), np.uint8),
        )

    def __len__(self) -> int:
        return len(self.unit)

    def evidence(self, i: int) -> str:
        return bytes(self.ev_bytes[self.ev_offsets[i] : self.ev_offsets[i + 1]]).decode()

    def unit_range(self, u: int, q: int | None = None) -> tuple[int, int]:
        if q is None:
            return int(self.offsets[u * NQ]), int(self.offsets[(u + 1) * NQ])
        return int(self.offsets[u * NQ + q]), int(self.offsets[u * NQ + q + 1])

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f).nbytes for f in self.__dataclass_fields__)


@dataclass
class QAStats:
    """Counts used by the sampling weights, keyed by names so that they carry over from one
    generator to another (compute them on train, pass them to the valid generator).

    n_units    concept -> number of report units where it is present
    n_answers  (template_id, concept, answer) -> number of candidates
    """

    n_units: pd.Series
    n_answers: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_units": self.n_units.to_dict(),
            "n_answers": [[*k, int(v)] for k, v in self.n_answers.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAStats:
        rows = data["n_answers"]
        index = pd.MultiIndex.from_tuples(
            [tuple(r[:3]) for r in rows], names=["template_id", "concept", "answer"]
        )
        return cls(
            n_units=pd.Series(data["n_units"], name="n_units", dtype=np.int64),
            n_answers=pd.Series([r[3] for r in rows], index=index, name="n", dtype=np.int64),
        )


_GENERATORS: dict[str, type[QAGenerator]] = {}


def register_generator(name: str):
    def deco(cls: type[QAGenerator]) -> type[QAGenerator]:
        if name in _GENERATORS:
            raise ValueError(f"generator {name!r} registered twice")
        cls.name = name
        _GENERATORS[name] = cls
        return cls

    return deco


def available_generators() -> dict[str, type[QAGenerator]]:
    return dict(_GENERATORS)


def get_generator(name: str) -> type[QAGenerator]:
    if name not in _GENERATORS:
        raise KeyError(f"unknown generator {name!r}; available: {sorted(_GENERATORS)}")
    return _GENERATORS[name]


@register_generator("generic")
class QAGenerator:
    """Dataset-independent Q/A generator. One item = one member of the exports (member_column);
    items of the same report unit share its candidates.

    Dataset subclasses override:
      member_column      column of the exports naming an item (e.g. VolumeName)
      item_table(store)  DataFrame [item, unit_id] of the items to train on
      normalize_query(q) turn a user query into an item name or prefix (resolve)
    """

    name: ClassVar[str] = ""
    member_column: ClassVar[str] = "id"

    def __init__(
        self,
        templates: str | Path = "chest_ct_findings",
        sampling_config: dict[str, Any] | SamplingConfig | None = None,
        overlay: str | Path | bool | None = None,
        member_column: str | None = None,
    ):
        """templates: Q/A template set (qa_templates/<name>) or folder.
        sampling_config: default config of sample_qa_from_item (over sampling.yaml).
        overlay: dataset overlay YAML. None = <template set>/datasets/<generator name>.yaml if it
        exists; False = none."""
        if member_column:
            self.member_column = member_column
        if overlay is None:
            default = resolve_spec_dir(templates) / "datasets" / f"{self.name}.yaml"
            overlay = default if default.exists() else None
        self.spec: QASpec = load_qa_spec(templates, overlay=overlay or None)
        self.template_ids = [t.id for t in self.spec.templates]
        self.options: list[list[str] | None] = self._check_templates()
        self.sampling_config = resolve_config(self.spec.sampling, sampling_config)
        self.connected = False
        self._states: dict[str, SamplingState] = {}

    # ---------------------------------------------------------------- templates
    def _check_templates(self) -> list[list[str] | None]:
        out = []
        for t in self.spec.templates:
            builder = get_builder(t.builder)
            opts = builder.options(t, self.spec.noul_options)
            if t.qtype == "noul" and opts is not None and len(opts) != 2:
                raise ValueError(f"template {t.id!r}: a noul template needs 2 options")
            if opts is not None and len(opts) < 2:
                raise ValueError(f"template {t.id!r}: needs at least 2 options")
            slots = {"finding": "x", **{s: "x" for s in builder.slots}}
            for text in self.spec.phrasings[t.id].texts:
                try:
                    text.format(**slots)
                except (KeyError, IndexError, ValueError) as e:
                    raise ValueError(
                        f"template {t.id!r}: phrasing {text!r} is not valid ({e!r}); "
                        f"placeholders available: {sorted(slots)}"
                    ) from None
            out.append(opts)
        return out

    # ---------------------------------------------------------------- dataset hooks
    def item_table(self, store: StructuredReportStore) -> pd.DataFrame:
        return store.members.rename(columns={store.member_column: "item"})[["item", "unit_id"]]

    def normalize_query(self, query: str) -> str:
        return query.strip()

    # ---------------------------------------------------------------- connect / compile
    def connect_structured_report(
        self, exports_dir: str | Path, stats: QAStats | None = None
    ) -> QAGenerator:
        """Read an export folder (rsx export) and compile the candidates. stats: counts of
        another generator (train) to weight this one's sampling with."""
        store = StructuredReportStore.from_exports(exports_dir, self.member_column)
        return self.connect_store(store, stats)

    def connect_store(
        self, store: StructuredReportStore, stats: QAStats | None = None
    ) -> QAGenerator:
        t0 = time.perf_counter()
        facts = build_facts(store, self.spec.concepts)
        t1 = time.perf_counter()
        frames = []
        for ti, t in enumerate(self.spec.templates):
            df = get_builder(t.builder).build(facts, t)
            df["template"] = ti
            df["qtype"] = QTYPES.index(t.qtype)
            frames.append(df)
        cand = pd.concat(frames, ignore_index=True)
        self.cand = Candidates.from_frame(cand, len(store.units))

        items = self.item_table(store)
        self.items = np.array(items["item"], dtype=str)
        self.item_unit = store.units.get_indexer(items["unit_id"]).astype(np.int32)
        self._item_pos = {name: i for i, name in enumerate(self.items)}
        self.units = store.units
        self.facts = facts
        self.concepts = facts.concepts
        self._side_labels = [
            [str(t.answer.get("sides", {}).get(s, s)) for s in SIDES] for t in self.spec.templates
        ]
        # distractor pool of choice_which_present: concepts answered No, per unit (CSR)
        neg = facts.state[facts.state["answer"] == 0].sort_values(["unit_idx", "concept_idx"])
        self.neg_concept = neg["concept_idx"].to_numpy(np.int32)
        self.neg_closed_world = (neg["flags"].to_numpy(np.uint8) & _CW) > 0
        self.neg_offsets = np.searchsorted(
            neg["unit_idx"].to_numpy(), np.arange(len(store.units) + 1)
        ).astype(np.int64)
        self.own_stats = self._compute_stats()
        self.stats = stats or self.own_stats
        self.source = getattr(store, "source", "")
        self.connected = True
        self._states = {}
        self._state = self._get_state(None)
        self.timings = {
            "facts_s": round(t1 - t0, 2),
            "compile_s": round(time.perf_counter() - t1, 2),
        }
        return self

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("call connect_structured_report(<exports folder>) first")

    def _compute_stats(self) -> QAStats:
        c = self.cand
        df = pd.DataFrame(
            {
                "template_id": np.array(self.template_ids, dtype=object)[c.template],
                "concept": self.concepts.to_numpy()[c.concept],
                "answer": c.answer.astype(int),
            }
        )
        n_answers = df.groupby(["template_id", "concept", "answer"]).size().rename("n")
        n_units = pd.Series(self.facts.n_units, index=self.concepts.to_numpy(), name="n_units")
        return QAStats(n_units=n_units, n_answers=n_answers)

    # ---------------------------------------------------------------- items
    def __len__(self) -> int:
        self._require_connected()
        return len(self.items)

    def resolve(self, query: str | int) -> list[str]:
        """Items matching a full name, an index, or a name prefix."""
        self._require_connected()
        if isinstance(query, int | np.integer):
            return [str(self.items[query])]
        q = self.normalize_query(str(query))
        if q in self._item_pos:
            return [q]
        return [str(x) for x in self.items[np.char.startswith(self.items, q)]]

    def item_index(self, query: str | int) -> int:
        """Index of one item; a prefix is accepted if all its items share one report."""
        if isinstance(query, int | np.integer):
            return int(query)
        matches = self.resolve(query)
        if not matches:
            raise KeyError(f"no item matches {query!r}")
        idx = [self._item_pos[m] for m in matches]
        if len({int(self.item_unit[i]) for i in idx}) > 1:
            raise KeyError(
                f"{query!r} matches items of different reports: {matches[:6]}"
                + (" ..." if len(matches) > 6 else "")
            )
        return idx[0]

    # ---------------------------------------------------------------- sampling
    def _get_state(self, config: dict[str, Any] | SamplingConfig | None) -> SamplingState:
        """Compiled state of the default config, or of the default config updated by `config`.
        The last few states are cached."""
        cfg = (
            self.sampling_config if config is None else resolve_config(self.sampling_config, config)
        )
        key = cfg.model_dump_json()
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= 8:
                self._states.pop(next(iter(self._states)))
            state = self._states[key] = compile_state(self, cfg)
        return state

    def set_sampling_config(
        self, config: dict[str, Any] | SamplingConfig | None = None, replace: bool = False
    ) -> SamplingConfig:
        """Change the default sampling config (e.g. between epochs, for a curriculum).
        config updates the current one; replace=True starts again from sampling.yaml."""
        base = self.spec.sampling if replace else self.sampling_config
        self.sampling_config = resolve_config(base, config)
        if self.connected:
            self._state = self._get_state(None)
        return self.sampling_config

    def sample_qa_from_item(
        self,
        item: str | int,
        k: int = 8,
        sampling_config: dict[str, Any] | SamplingConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> list[QA]:
        """Draw up to k questions about one item (name, prefix or index), without repeating a
        candidate. sampling_config updates the default config for this call (compiled once,
        cached). rng: a numpy Generator or a seed (None: fresh entropy)."""
        self._require_connected()
        state = self._state if sampling_config is None else self._get_state(sampling_config)
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        i = self.item_index(item)
        picks = self._draw(int(self.item_unit[i]), k, state, rng)
        return [self._render(int(j), i, state, rng) for j in picks]

    def _draw(self, u: int, k: int, st: SamplingState, rng: np.random.Generator) -> np.ndarray:
        c = self.cand
        cap = st.npos[u * NQ : (u + 1) * NQ]
        p = st.p_qtype * (cap > 0)
        counts = np.zeros(NQ, np.int64)
        remaining = k
        while remaining > 0 and p.sum() > 0:  # a qtype short of candidates: spread the rest
            take = np.minimum(rng.multinomial(remaining, p / p.sum()), cap - counts)
            counts += take
            remaining -= int(take.sum())
            p = p * (counts < cap)
        picks = []
        for q in np.flatnonzero(counts):
            a, b = int(c.offsets[u * NQ + q]), int(c.offsets[u * NQ + q + 1])
            m = int(counts[q])
            keys = st.logw[a:b] + rng.gumbel(size=b - a)  # Gumbel top-k = weighted, no repeat
            top = (
                np.flatnonzero(np.isfinite(keys))
                if m >= b - a
                else np.argpartition(-keys, m - 1)[:m]
            )
            picks.append(a + top)
        picks = np.concatenate(picks) if picks else np.zeros(0, np.int64)
        if st.accept is not None:  # answer_balance.drop_excess
            picks = picks[rng.random(len(picks)) < st.accept[picks]]
        if remaining > 0 and st.config.fill == "rephrase" and len(picks):
            a, b = c.unit_range(u)
            pool = a + np.flatnonzero(st.w[a:b] > 0)
            w = st.w[pool]
            picks = np.concatenate([picks, rng.choice(pool, remaining, p=w / w.sum())])
        return rng.permutation(picks)

    def _render(self, j: int, item: int, st: SamplingState, rng: np.random.Generator) -> QA:
        c = self.cand
        t = int(c.template[j])
        tpl = self.spec.templates[t]
        ps = self.spec.phrasings[tpl.id]
        allowed = st.phrasings[t]
        p = int(allowed[rng.integers(len(allowed))])
        concept = int(c.concept[j])
        names = st.displays[concept]
        slots = {"finding": names[rng.integers(len(names))]}
        if c.slot[j] >= 0:
            slots["side"] = self._side_labels[t][c.slot[j]]
        flags = int(c.flags[j]) | (_LOW if c.fuzzy[j] < st.config.low_evidence_threshold else 0)
        options = self.options[t]
        if options is None:  # choice_which_present: the present concept + negated ones
            u = int(c.unit[j])
            a, b = int(self.neg_offsets[u]), int(self.neg_offsets[u + 1])
            pool = a + np.flatnonzero(st.distractor_w[a:b] > 0)
            dw = st.distractor_w[pool]
            chosen = rng.choice(pool, (tpl.n_options or 4) - 1, replace=False, p=dw / dw.sum())
            if self.neg_closed_world[chosen].any():
                flags |= _CW
            shown = [concept, *self.neg_concept[chosen].tolist()]
            perm = rng.permutation(len(shown))
            labels = [st.displays[x][rng.integers(len(st.displays[x]))] for x in shown]
            options = [labels[x] for x in perm]
            answer = int(np.flatnonzero(perm == 0)[0])
        elif tpl.qtype == "choice" and st.config.shuffle_choice_options:
            perm = rng.permutation(len(options))
            options = [options[x] for x in perm]
            answer = int(np.flatnonzero(perm == c.answer[j])[0])
        else:
            options = list(options)
            answer = int(c.answer[j])
        return QA(
            question=ps.texts[p].format(**slots),
            qtype=tpl.qtype,
            options=options,
            answer=answer,
            item=str(self.items[item]),
            template_id=tpl.id,
            phrasing_id=ps.ids[p],
            concept=str(self.concepts[concept]),
            flags=flag_names(flags),
            evidence=c.evidence(j),
        )

    def item_weights(
        self, sampling_config: dict[str, Any] | SamplingConfig | None = None
    ) -> np.ndarray:
        """One weight per item (aligned with self.items), for WeightedRandomSampler: the rarity
        weight of its rarest present concept (mode max, at least 1) or their sum (mode sum).
        0 for items without any drawable candidate."""
        self._require_connected()
        st = self._state if sampling_config is None else self._get_state(sampling_config)
        c = self.cand
        n_units = len(self.units)
        keep = (st.w > 0) & c.present
        pairs = np.unique(c.unit[keep].astype(np.int64) * len(self.concepts) + c.concept[keep])
        pu, pc = pairs // len(self.concepts), pairs % len(self.concepts)
        if st.config.item_weights.mode == "sum":
            value = np.bincount(pu, weights=st.rarity[pc], minlength=n_units)
        else:
            value = np.zeros(n_units)
            np.maximum.at(value, pu, st.rarity[pc])
        drawable = st.npos.reshape(n_units, NQ).sum(axis=1) > 0
        unit_w = np.where(drawable, np.maximum(value, 1.0), 0.0)
        return unit_w[self.item_unit]

    def freeze(
        self,
        items: Iterable[str | int] | None = None,
        k: int = 8,
        seed: int = 0,
        sampling_config: dict[str, Any] | SamplingConfig | None = None,
    ) -> pd.DataFrame:
        """A fixed Q/A set (e.g. for validation): k questions per item, the item's own seed
        derived from (seed, item index). One row per question; save it with to_parquet."""
        self._require_connected()
        idx = range(len(self.items)) if items is None else [self.item_index(x) for x in items]
        rows = []
        for i in idx:
            rng = np.random.default_rng([seed, i])
            for n, qa in enumerate(self.sample_qa_from_item(i, k, sampling_config, rng)):
                rows.append({"item_idx": i, "qa_idx": n, **qa.to_dict()})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- inspection
    def option_labels(self, template: int) -> list[str] | None:
        return self.options[template]

    def concept_display(self, concept: int) -> list[str]:
        """Display names of a concept (all splits)."""
        name = str(self.concepts[concept])
        if name.startswith("other:"):
            return [name.split(":", 1)[1]]
        ps = self.spec.displays.get(name)
        return list(ps.texts) if ps else [name.replace("_", " ")]

    def stats_table(self, by_concept: bool = True) -> pd.DataFrame:
        """Candidate counts per template (and concept) and answer, with answer labels."""
        self._require_connected()
        s = self.own_stats.n_answers.reset_index()
        tpl_pos = {t: i for i, t in enumerate(self.template_ids)}

        def label(row) -> str:
            opts = self.options[tpl_pos[row.template_id]]
            return "(drawn)" if opts is None or row.answer < 0 else opts[row.answer]

        s["qtype"] = [self.spec.templates[tpl_pos[t]].qtype for t in s["template_id"]]
        s["answer_label"] = [label(r) for r in s.itertuples()]
        keys = ["qtype", "template_id", "concept"] if by_concept else ["qtype", "template_id"]
        out = s.groupby([*keys, "answer", "answer_label"], sort=False)["n"].sum().reset_index()
        out["_t"] = out["template_id"].map(tpl_pos)
        out = out.sort_values(["_t", *keys[2:], "answer"]).drop(columns="_t")
        return out.reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """One row per template: candidates, units and concepts covered, answer distribution."""
        self._require_connected()
        c = self.cand
        rows = []
        for ti, t in enumerate(self.spec.templates):
            m = c.template == ti
            answers = pd.Series(c.answer[m]).value_counts().sort_index()
            opts = self.options[ti]
            dist = ", ".join(
                f"{'drawn' if opts is None or a < 0 else opts[a]}: {n}" for a, n in answers.items()
            )
            n_concepts = len(np.unique(c.concept[m]))
            rows.append(
                {
                    "template_id": t.id,
                    "qtype": t.qtype,
                    "candidates": int(m.sum()),
                    "units": len(np.unique(c.unit[m])),
                    "concepts": n_concepts,
                    "phrasings": len(self.spec.phrasings[t.id].texts),
                    "answers": dist,
                }
            )
        return pd.DataFrame(rows)

    def describe(self) -> dict[str, Any]:
        self._require_connected()
        flag_counts = {
            name: int(((self.cand.flags & (1 << i)) > 0).sum()) for i, name in enumerate(FLAGS[:-1])
        }
        return {
            "generator": self.name,
            "templates": str(self.spec.path),
            "overlay": str(self.spec.overlay) if self.spec.overlay else None,
            "exports": self.source,
            "items": len(self.items),
            "units": len(self.units),
            "concepts": len(self.concepts),
            "candidates": len(self.cand),
            "candidates_MB": round(self.cand.nbytes / 2**20, 1),
            "flagged_candidates": flag_counts,
            "negation_conflicts": self.facts.n_conflicts,
            **self.timings,
        }
