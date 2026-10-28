from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from rsx.qa import CTRATEGenerator, QAGenerator, load_qa_spec
from rsx.qa.facts import FLAG_BIT
from rsx.qa.templates import QA_TEMPLATES_DIR, Holdout, Phrasing, assign_splits


@pytest.fixture
def gen(qa_exports: Path) -> CTRATEGenerator:
    return CTRATEGenerator("chest_ct_findings", overlay=False).connect_structured_report(qa_exports)


def _template_dir(tmp_path: Path, edit) -> Path:
    d = tmp_path / "tpl"
    shutil.copytree(QA_TEMPLATES_DIR / "chest_ct_findings", d)
    data = yaml.safe_load((d / "templates.yaml").read_text())
    edit(data)
    (d / "templates.yaml").write_text(yaml.safe_dump(data))
    return d


def _candidates(gen, template_id: str, unit: str) -> list[tuple[str, int, int]]:
    """(concept, slot, answer) of one template in one unit."""
    c = gen.cand
    ti = gen.template_ids.index(template_id)
    a, b = c.unit_range(gen.units.get_loc(unit))
    return [
        (str(gen.concepts[c.concept[i]]), int(c.slot[i]), int(c.answer[i]))
        for i in range(a, b)
        if c.template[i] == ti
    ]


def test_items_share_unit(gen):
    assert len(gen) == 4
    i1, i2 = gen.item_index("valid_1_a_1.nii.gz"), gen.item_index("valid_1_a_2.nii.gz")
    assert i1 != i2 and gen.item_unit[i1] == gen.item_unit[i2]


def test_ct_rate_resolve(gen):
    assert gen.resolve("valid_01") == ["valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz"]  # not valid_10
    assert gen.resolve("valid_1_a_2") == ["valid_1_a_2.nii.gz"]
    assert gen.item_index("valid_0001_a") == gen.item_index("valid_1_a_1.nii.gz")
    with pytest.raises(KeyError, match="no item"):
        gen.item_index("valid_99")
    with pytest.raises(KeyError, match="different reports"):
        gen.item_index("valid_")


def test_candidates(gen):
    pres = _candidates(gen, "noul_presence", "u1")
    assert ("pleural_effusion", -1, 0) in pres and ("lung_nodule", -1, 1) in pres
    assert not any(c == "lung_mass" for c, _, _ in pres)
    side = _candidates(gen, "noul_presence_side", "u1")
    assert ("pneumothorax", 0, 0) in side and ("pneumothorax", 1, 0) not in side
    lat = {c: a for c, _, a in _candidates(gen, "choice_laterality", "u1")}
    lat_options = gen.options[gen.template_ids.index("choice_laterality")]
    assert lat_options[lat["lung_nodule"]] == "both sides"
    # lingula is mapped to the left upper lobe
    lobe = _candidates(gen, "choice_lobe", "u10")
    assert gen.options[gen.template_ids.index("choice_lobe")][lobe[0][2]] == "left upper lobe"
    # 6 mm lies on an edge: bin [6, 8)
    size = _candidates(gen, "score_nodule_size", "u10")
    assert gen.options[gen.template_ids.index("score_nodule_size")][size[0][2]] == "6 to 8 mm"
    severity = _candidates(gen, "score_severity", "u2")
    assert not [c for c in severity if c[0] == "pleural_effusion"]  # severities disagree
    # u1 has 3 usable negations: one which_present candidate per present concept
    which = _candidates(gen, "choice_which_present", "u1")
    assert {c for c, _, _ in which} == {"lung_nodule", "emphysema", "ground_glass_opacity"}
    assert _candidates(gen, "choice_which_present", "u10") == []  # 2 negations only


def test_flags_propagate(gen):
    c = gen.cand
    ti = gen.template_ids.index("choice_laterality")
    nod = gen.concepts.get_loc("lung_nodule")
    m = (c.template == ti) & (c.concept == nod) & (c.unit == gen.units.get_loc("u1"))
    assert (c.flags[m] & FLAG_BIT["location_alert"]).all()
    assert c.fuzzy[m].min() == pytest.approx(80.0)


def test_candidate_offsets(gen):
    c = gen.cand
    for u in range(len(gen.units)):
        a, b = c.unit_range(u)
        assert (c.unit[a:b] == u).all()
        for q in range(3):
            qa, qb = c.unit_range(u, q)
            assert (c.qtype[qa:qb] == q).all()
    assert c.evidence(0)


def test_summary_and_stats(gen):
    s = gen.summary()
    assert list(s["template_id"]) == gen.template_ids
    table = gen.stats_table()
    assert {"template_id", "concept", "answer_label", "n"} <= set(table.columns)
    assert gen.stats.n_units["lung_nodule"] == 2
    round_trip = type(gen.stats).from_dict(gen.stats.to_dict())
    assert round_trip.n_answers.equals(gen.stats.n_answers)


def test_not_connected():
    with pytest.raises(RuntimeError, match="connect_structured_report"):
        QAGenerator("chest_ct_findings").summary()


def test_template_validation(tmp_path: Path):
    def bad_slot(data):
        data["templates"][0]["phrasings"] = ["Is there {lesion}?"]

    with pytest.raises(ValueError, match="placeholders available"):
        QAGenerator(_template_dir(tmp_path, bad_slot))

    def bad_builder(data):
        data["templates"][0]["builder"] = "nope"

    shutil.rmtree(tmp_path / "tpl")
    with pytest.raises(KeyError, match="unknown builder"):
        QAGenerator(_template_dir(tmp_path, bad_builder))


def test_unknown_where_column(tmp_path: Path, qa_exports: Path):
    def edit(data):
        data["templates"][0]["where"] = {"colour": ["red"]}

    gen = QAGenerator(_template_dir(tmp_path, edit), member_column="VolumeName")
    with pytest.raises(ValueError, match="colour"):
        gen.connect_structured_report(qa_exports)


def test_phrasing_splits():
    ps = [Phrasing(text=f"question {i}?") for i in range(20)]
    s1 = assign_splits("t", ps, Holdout(fraction=0.3))
    assert s1 == assign_splits("t", ps, Holdout(fraction=0.3))  # deterministic
    assert 0 < s1.count("heldout") < 20
    # adding phrasings does not move the existing ones
    assert assign_splits("t", [*ps, Phrasing(text="new?")], Holdout(fraction=0.3))[:20] == s1
    assert assign_splits("t", ps[:1], Holdout(fraction=0.99)) == ["train"]  # one stays train
    forced = [Phrasing(text="a?", split="heldout"), Phrasing(text="b?")]
    assert assign_splits("t", forced, Holdout(fraction=0.0)) == ["heldout", "train"]
    spec = load_qa_spec("chest_ct_findings")
    assert all("train" in p.splits for p in spec.phrasings.values())
    assert spec.phrasing_table()["split"].isin(["train", "heldout"]).all()


# ---------------------------------------------------------------- sampling

ONLY_WHICH = {"which_present": {"min_negated": 0}}


def _many(gen, item, cfg=None, n=200, k=8):
    rng = np.random.default_rng(0)
    return [q for _ in range(n) for q in gen.sample_qa_from_item(item, k, cfg, rng)]


def test_sample_basic(gen):
    qas = gen.sample_qa_from_item("valid_1_a_1.nii.gz", 5, rng=0)
    assert 0 < len(qas) <= 5
    assert len({(q.template_id, q.concept, q.question) for q in qas}) == len(qas)
    for q in qas:
        assert q.item == "valid_1_a_1.nii.gz" and 0 <= q.answer < len(q.options)
    again = gen.sample_qa_from_item("valid_1_a_1.nii.gz", 5, rng=0)
    assert [q.to_dict() for q in again] == [q.to_dict() for q in qas]


def test_score_options_keep_order(gen):
    tpl = gen.template_ids.index("score_nodule_size")
    for q in _many(gen, "valid_10_a_1.nii.gz"):
        if q.template_id == "score_nodule_size":
            assert q.ordinal and q.options == gen.options[tpl]
            assert q.answer_text == "6 to 8 mm"
        if q.template_id == "choice_lobe":
            assert sorted(q.options) == sorted(gen.options[gen.template_ids.index("choice_lobe")])
            assert q.answer_text == "left upper lobe"


def test_qtype_and_template_weights(gen):
    qas = _many(gen, "valid_1_a_1.nii.gz", {"qtype": {"noul": 1, "choice": 0, "score": 0}}, n=20)
    assert {q.qtype for q in qas} == {"noul"}
    off = {"templates": {"noul_presence": 0}}
    assert "noul_presence" not in {q.template_id for q in _many(gen, "valid_1_a_1.nii.gz", off)}
    with pytest.raises(ValueError, match="unknown template"):
        gen.sample_qa_from_item(0, 2, {"templates": {"nope": 1}})
    with pytest.raises(ValueError):
        gen.sample_qa_from_item(0, 2, {"qtyp": {}})


def test_flag_weights(gen):
    flagged = [q for q in _many(gen, "valid_1_a_1.nii.gz") if "location_alert" in q.flags]
    assert flagged  # drawn by default
    cfg = {"flag_weights": {"location_alert": 0}}
    assert not [q for q in _many(gen, "valid_1_a_1.nii.gz", cfg) if "location_alert" in q.flags]
    # the u1 nodule evidence_fuzzy is 80: low_evidence with the default threshold 90
    assert any("low_evidence" in q.flags for q in flagged)


def test_phrasing_split(gen):
    splits = gen.spec.phrasing_table().set_index("phrasing_id")["split"]
    train = _many(gen, "valid_1_a_1.nii.gz", n=50)
    assert {splits[q.phrasing_id] for q in train} == {"train"}
    held = _many(gen, "valid_1_a_1.nii.gz", {"phrasings": "heldout"}, n=50)
    assert held and {splits[q.phrasing_id] for q in held} == {"heldout"}


def test_which_present_distractors(gen):
    shown = set()
    for q in _many(gen, "valid_1_a_1.nii.gz", ONLY_WHICH):
        if q.template_id == "choice_which_present":
            assert len(q.options) == 4 and q.answer_text in gen.concept_display(
                gen.concepts.get_loc(q.concept)
            )
            shown |= set(q.options) - {q.answer_text}
    negated = {"pleural_effusion", "lymphadenopathy", "cardiomegaly"}
    allowed = {n for c in negated for n in gen.concept_display(gen.concepts.get_loc(c))}
    assert shown and shown <= allowed
    # default min_negated (20) is not reached in this tiny export
    assert "choice_which_present" not in {q.template_id for q in _many(gen, 0, n=30)}


def test_item_weights_and_freeze(gen):
    w = gen.item_weights()
    assert w.shape == (len(gen),) and (w >= 1).all()
    assert w[0] == w[1]  # two reconstructions of one report
    a = gen.freeze(k=4, seed=3)
    b = gen.freeze(k=4, seed=3)
    assert a.equals(b) and set(a["item"]) == set(gen.items)
    assert {"question", "options", "answer", "ordinal", "flags"} <= set(a.columns)


def test_drop_excess_and_blind(gen):
    from rsx.qa.diagnostics import blind_baseline

    qas = _many(gen, 0, {"answer_balance": {"drop_excess": True}}, n=20)
    assert all(0 <= q.answer < len(q.options) for q in qas)
    table = blind_baseline(gen, n_items=50, k=4)
    assert {"template_id", "blind_accuracy", "chance"} <= set(table.columns)


def test_set_sampling_config(gen):
    gen.set_sampling_config({"qtype": {"noul": 0, "choice": 1, "score": 0}})
    assert {q.qtype for q in _many(gen, 0, n=10)} <= {"choice"}
    gen.set_sampling_config(replace=True)
    assert gen.sampling_config.qtype == {"noul": 0.4, "choice": 0.3, "score": 0.3}


def test_overlay(tmp_path: Path, qa_exports: Path):
    ct = CTRATEGenerator("chest_ct_findings")
    plain = CTRATEGenerator("chest_ct_findings", overlay=False)
    sev = "score_severity"
    assert ct.options[ct.template_ids.index(sev)][-1] == "moderate or severe"
    assert plain.options[plain.template_ids.index(sev)][-1] == "severe"
    assert QAGenerator("chest_ct_findings").spec.overlay is None  # generic: no ct_rate overlay

    overlay = tmp_path / "cw.yaml"
    overlay.write_text("concepts: {closed_world: [pleural_effusion, lung_nodule]}\n")
    gen = CTRATEGenerator("chest_ct_findings", overlay=overlay).connect_structured_report(
        qa_exports
    )
    pres = _candidates(gen, "noul_presence", "u10")
    assert ("pleural_effusion", -1, 0) in pres  # not mentioned -> No
    side = _candidates(gen, "noul_presence_side", "u10")
    assert ("lung_nodule", 0, 0) in side  # nodule only on the left -> right No
    cw = [q for q in _many(gen, "valid_10_a_1.nii.gz") if "closed_world" in q.flags]
    assert cw and all(q.answer_text == "No" for q in cw)
    off = {"flag_weights": {"closed_world": 0}}
    assert not [q for q in _many(gen, "valid_10_a_1.nii.gz", off) if "closed_world" in q.flags]
    bad = tmp_path / "bad.yaml"
    bad.write_text("templates: {nope: {qtype: noul}}\n")
    with pytest.raises(ValueError, match="unknown template"):
        CTRATEGenerator("chest_ct_findings", overlay=bad)
