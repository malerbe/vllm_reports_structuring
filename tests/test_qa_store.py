from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsx.qa import StructuredReportStore, load_qa_spec
from rsx.qa.facts import FLAG_BIT, build_facts, normalize_name


@pytest.fixture
def facts(qa_exports: Path):
    store = StructuredReportStore.from_exports(qa_exports)
    return store, build_facts(store, load_qa_spec("chest_ct_findings").concepts)


def _row(df: pd.DataFrame, store, facts, unit: str, concept: str, **kw) -> pd.DataFrame:
    if concept not in facts.concepts:
        return df.iloc[:0]
    u = store.units.get_loc(unit)
    c = facts.concepts.get_loc(concept)
    m = (df["unit_idx"] == u) & (df["concept_idx"] == c)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def test_store_dedup_and_lists(qa_exports: Path):
    store = StructuredReportStore.from_exports(qa_exports)
    assert list(store.units) == ["u1", "u10", "u2"]  # ufail dropped (status failed)
    assert len(store.members) == 4
    f = store.findings
    assert len(f) == 12  # one row per (unit, item), not per member
    assert f.loc[f["unit_id"] == "u10", "lobes"].iloc[0] == ("lingula",)
    assert store.negations["regions"].iloc[0] == ()  # parsed from the CSV fallback


def test_missing_table(tmp_path: Path):
    with pytest.raises(ValueError, match="missing export table"):
        StructuredReportStore({"reports": pd.DataFrame({"VolumeName": [], "unit_id": []})})


def test_state(facts):
    store, fx = facts
    st = fx.state

    def answer(unit, concept):
        r = _row(st, store, fx, unit, concept)
        return None if r.empty else int(r["answer"].iloc[0])

    assert answer("u1", "lung_nodule") == 1
    assert answer("u1", "pleural_effusion") == 0
    assert answer("u1", "cardiomegaly") == 0  # normal_statement
    assert answer("u1", "lymphadenopathy") == 0  # neutral qualifier
    assert answer("u1", "lung_mass") is None  # 'suspicious' narrows the negation
    assert answer("u1", "lung_opacity") is None  # blocked by ground-glass (implies)
    assert answer("u1", "pneumothorax") is None  # right side only: no generic answer
    assert answer("u2", "pleural_effusion") == 1  # present wins over the negation
    assert answer("u2", "consolidation") is None  # resolved: not present
    assert answer("u10", "pleural_effusion") is None  # not mentioned: unknown
    assert fx.n_conflicts == 2  # u2 pleural_effusion + u1 lung_opacity


def test_side_state(facts):
    store, fx = facts
    ss = fx.side_state

    def answer(unit, concept, side):
        r = _row(ss, store, fx, unit, concept, side=side)
        return None if r.empty else int(r["answer"].iloc[0])

    assert answer("u1", "pneumothorax", "right") == 0
    assert answer("u1", "pneumothorax", "left") is None
    assert answer("u2", "pneumothorax", "left") == 0  # generic negation: both sides
    assert answer("u1", "lung_nodule", "right") == 1
    assert answer("u1", "lung_nodule", "left") == 1
    assert answer("u1", "ground_glass_opacity", "left") == 1  # bilateral
    assert answer("u1", "emphysema", "left") is None  # side not given
    assert answer("u1", "pleural_effusion", "left") == 0


def test_agg(facts):
    store, fx = facts
    nod = _row(fx.agg, store, fx, "u1", "lung_nodule").iloc[0]
    assert nod["laterality"] == "bilateral"
    assert pd.isna(nod["lobe"])  # two lobes
    assert pd.isna(nod["count"])  # two top-level entries
    assert nod["flags"] & FLAG_BIT["location_alert"]
    assert nod["fuzzy"] == 80.0
    emph = _row(fx.agg, store, fx, "u1", "emphysema").iloc[0]
    assert emph["severity"] == "mild" and pd.isna(emph["laterality"])
    pe = _row(fx.agg, store, fx, "u2", "pleural_effusion").iloc[0]
    assert pd.isna(pe["severity"])  # minimal vs moderate
    assert pe["laterality"] == "bilateral"
    assert pe["size_mm_max"] == 15.0 and pe["size_measure"] == "thickness"
    single = _row(fx.agg, store, fx, "u10", "lung_nodule").iloc[0]
    assert single["lobe"] == "lingula" and single["count"] == "single"


def test_normalize_name():
    assert normalize_name("Accessory spleens") == "accessory spleen"
    assert normalize_name("the  Schmorl nodules.") == "schmorl nodule"
    assert normalize_name("budding tree appearances") == "budding tree appearance"
    assert normalize_name("osteoporosis") == "osteoporosis"
    assert normalize_name("bullous changes") == "bullous change"


def _answer(df: pd.DataFrame, store, fx, unit: str, concept: str, **kw) -> int | None:
    r = _row(df, store, fx, unit, concept, **kw)
    return None if r.empty else int(r["answer"].iloc[0])


def _facts_with(qa_exports: Path, tmp_path: Path, overlay: str):
    p = tmp_path / "overlay.yaml"
    p.write_text(overlay)
    store = StructuredReportStore.from_exports(qa_exports)
    return store, build_facts(store, load_qa_spec("chest_ct_findings", overlay=p).concepts)


def test_rules_and_exclude(facts):
    store, fx = facts
    st = fx.state
    # lymph_node_not_enlarged: its negations and its findings become No of lymphadenopathy
    assert _answer(st, store, fx, "u10", "lymphadenopathy") == 0
    assert _answer(st, store, fx, "u2", "lymphadenopathy") == 0
    assert "lymph_node_not_enlarged" not in fx.concepts and "infection" not in fx.concepts
    # 'vascular structures normal' is not a No of arterial wall calcification
    assert _answer(st, store, fx, "u2", "arterial_wall_calcification") is None
    # crazy paving adds a present septal thickening, flagged derived, without the GGO severity
    sept = _row(fx.agg, store, fx, "u2", "interlobular_septal_thickening").iloc[0]
    assert sept["flags"] & FLAG_BIT["derived"] and pd.isna(sept["severity"])
    assert sept["laterality"] == "bilateral"


def test_closed_world(qa_exports: Path, tmp_path: Path):
    store, fx = _facts_with(
        qa_exports,
        tmp_path,
        "concepts:\n"
        "  closed_world: [lung_mass, pleural_effusion, pneumothorax]\n"
        "  closed_world_if: {hepatic_steatosis: {regions: [liver]}}\n",
    )
    st, ss = fx.state, fx.side_state

    def cw(unit, concept):
        r = _row(st, store, fx, unit, concept)
        return (
            None
            if r.empty
            else (int(r["answer"].iloc[0]), bool(r["flags"].iloc[0] & FLAG_BIT["closed_world"]))
        )

    assert cw("u2", "lung_mass") == (0, True)  # not mentioned -> No
    assert cw("u1", "lung_mass") is None  # only 'no SUSPICIOUS mass': unknown
    assert cw("u10", "lung_mass") is None  # uncertain: unknown
    assert cw("u10", "pleural_effusion") == (0, True)
    assert cw("u1", "pleural_effusion") == (0, False)  # explicit No stays explicit
    assert cw("u2", "pleural_effusion") == (1, False)
    assert cw("u1", "pneumothorax") == (0, True)  # right negated, left silent
    assert _answer(ss, store, fx, "u1", "pneumothorax", side="right") == 0
    assert _answer(ss, store, fx, "u10", "lung_mass", side="left") is None
    # conditional: a liver entry is needed, and no unnamed (`other`) liver finding
    assert cw("u10", "hepatic_steatosis") == (0, True)
    assert cw("u2", "hepatic_steatosis") is None
    assert cw("u1", "hepatic_steatosis") is None


def _store(findings: list[dict], negations: list[dict]) -> StructuredReportStore:
    from conftest import _finding, _negation  # noqa: F401  (row helpers)

    units = sorted({r["unit_id"] for r in findings + negations})
    reports = pd.DataFrame(
        [{"VolumeName": f"{u}_1.nii.gz", "unit_id": u, "status": "ok"} for u in units]
    )

    def per(rows):
        return pd.DataFrame([{"VolumeName": f"{r['unit_id']}_1.nii.gz", **r} for r in rows])

    return StructuredReportStore(
        {"reports": reports, "findings": per(findings), "negations": per(negations)}
    )


def _spec(**update):
    base = load_qa_spec("chest_ct_findings").concepts
    return type(base).model_validate({**base.model_dump(), **update})


def test_side_closed_world_edge_cases():
    from conftest import _finding, _negation

    store = _store(
        [
            # lung_opacity right only + consolidation left: left lung_opacity must stay unknown
            _finding("a", 0, "lung_opacity", laterality="right"),
            _finding("a", 1, "consolidation", laterality="left"),
            # nodule right + 'no suspicious nodule in the left lung': left unknown
            _finding("b", 0, "lung_nodule", laterality="right"),
            # kidney lesion right, no kidney entry condition met? (the finding itself is one)
            _finding("c", 0, "kidney_lesion", region="kidney", laterality="right"),
            _finding("c", 1, "other", name="renal thing", region="kidney", laterality="left"),
            # uncertain consolidation: lung_opacity (parent) unknown
            _finding("d", 0, "consolidation", status="uncertain"),
        ],
        [
            _negation("b", 0, "lung_nodule", laterality="left", qualifier="suspicious"),
            _negation("e", 0, "pleural_effusion"),
        ],
    )
    spec = _spec(
        closed_world=["lung_opacity", "lung_nodule"],
        closed_world_if={"kidney_lesion": {"regions": ["kidney"]}},
    )
    fx = build_facts(store, spec)
    ss, st = fx.side_state, fx.state

    def side(unit, concept, s):
        return _answer(ss, store, fx, unit, concept, side=s)

    assert side("a", "lung_opacity", "right") == 1
    assert side("a", "lung_opacity", "left") is None  # consolidation on the left
    assert side("b", "lung_nodule", "left") is None  # narrowed lateral negation
    assert side("b", "lung_nodule", "right") == 1
    assert side("c", "kidney_lesion", "left") is None  # `other` kidney finding blocks the condition
    assert _answer(st, store, fx, "d", "lung_opacity") is None  # uncertain child
    assert _answer(st, store, fx, "e", "lung_opacity") == 0  # silence: closed world No
    assert not st.duplicated(["unit_idx", "concept_idx"]).any()
    assert not ss.duplicated(["unit_idx", "concept_idx", "side"]).any()


def test_finding_rules_sources():
    from conftest import _finding, _negation

    store = _store(
        [
            _finding(
                "a", 0, "ground_glass_opacity", change="resolved", evidence="Crazy paving resolved."
            ),
            _finding("b", 0, "lymph_node_not_enlarged", region="mediastinum", size_mm_min=12.0),
            _finding("c", 0, "lymph_node_not_enlarged", region="mediastinum", size_mm_min=6.0),
            _finding("d", 0, "lymph_node_not_enlarged", region="mediastinum", status="uncertain"),
        ],
        [_negation("e", 0, "pleural_effusion")],
    )
    fx = build_facts(store, _spec())
    st = fx.state
    assert "interlobular_septal_thickening" not in fx.concepts  # not from a resolved finding
    assert _answer(st, store, fx, "b", "lymphadenopathy") is None  # short axis 12 mm
    assert _answer(st, store, fx, "c", "lymphadenopathy") == 0
    assert _answer(st, store, fx, "d", "lymphadenopathy") is None  # not from an uncertain one


def test_closed_world_lists_validated():
    from rsx.qa.templates import ConceptSpec

    with pytest.raises(ValueError, match="duplicated"):
        ConceptSpec(closed_world=["emphysema", "emphysema"])
    with pytest.raises(ValueError, match="in both"):
        ConceptSpec(
            closed_world=["liver_lesion"], closed_world_if={"liver_lesion": {"regions": ["liver"]}}
        )
