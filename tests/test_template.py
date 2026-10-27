from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from conftest import RADFIELDS_TEMPLATES, REPO, valid_report
from pydantic import ValidationError

from rsx.template import (
    ListOf,
    Section,
    TemplateError,
    TemplateWarning,
    TextLeaf,
    TypedLeaf,
    build_model,
    check_schema_portability,
    field_catalog,
    load_template,
    parse_instructions,
    render_reader_view,
    to_field_name,
    vllm_json_schema,
)


def test_radfields_semantics():
    root = parse_instructions(
        {"a": "hint", "b": None, "c": 0.0, "d": [], "e": [{"x": "y"}], "f": {"g": ""}}
    )
    assert root.fields["a"] == TextLeaf("hint")
    assert root.fields["b"] == TextLeaf("")
    assert root.fields["c"] == TextLeaf("0.0")
    assert root.fields["d"] == ListOf(TextLeaf(""))
    assert isinstance(root.fields["e"], ListOf) and isinstance(root.fields["e"].item, Section)
    assert isinstance(root.fields["f"], Section)


def test_typed_leaves_and_nullability():
    root = parse_instructions(
        {
            "s": {"_type": "enum", "values": ["a", "b"]},
            "n": {"_type": "number", "_nullable": False},
            "l": [{"_type": "integer"}],
            "sec": {"_nullable": True, "_hint": "h", "x": "y"},
        }
    )
    assert root.fields["s"] == TypedLeaf("enum", "", True, ("a", "b"))
    assert root.fields["n"].nullable is False
    assert root.fields["l"].item.nullable is False  # list items: not nullable by default
    assert root.fields["sec"].nullable and root.fields["sec"].hint == "h"


@pytest.mark.parametrize(
    "raw, msg",
    [
        ({"a": {"_type": "date"}}, "_type must be one of"),
        ({"a": {"_type": "enum"}}, "non-empty list"),
        ({"a": {"_type": "enum", "values": ["x", True]}}, "must be strings"),
        ({"a": {"_type": "enum", "values": ["x", "x"]}}, "duplicate"),
        ({"a": {"_type": "number", "values": ["x"]}}, "only allowed for _type: enum"),
        ({"a": {"_type": "number", "_hnt": "x"}}, "unknown key"),
        ({"a": {"_hint": "x"}}, "empty section"),
        ({"a": {"_required": True, "b": "c"}}, "unknown reserved key"),
        ({"a": {"_type": "number", "_nullable": "yes"}}, "_nullable must be"),
        ({True: "x"}, "keys must be strings"),
    ],
)
def test_template_errors_carry_path(raw, msg):
    with pytest.raises(TemplateError, match=msg) as e:
        parse_instructions(raw)
    assert "$" in str(e.value)


def test_heterogeneous_list_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parse_instructions({"l": [{"a": "x", "b": "y"}, {"a": "x"}]})
    assert any(issubclass(w.category, TemplateWarning) for w in caught)


def test_field_names_and_aliases():
    assert to_field_name("länge_mm") == "laenge_mm"
    assert to_field_name("1st") == "f_1st"
    assert to_field_name("json") == "f_json"
    assert to_field_name("class") == "f_class"
    root = parse_instructions({"länge mm": "x", "json": "y"})
    model = build_model(root)
    obj = model.model_validate({"länge mm": "1", "json": "2"})
    assert obj.model_dump(by_alias=True) == {"länge mm": "1", "json": "2"}


def test_field_name_collision():
    with pytest.raises(TemplateError, match="collide"):
        build_model(parse_instructions({"a b": "x", "a_b": "y"}))


def test_model_validation(typed_template_dir: Path):
    t = load_template(typed_template_dir)
    t.model.model_validate(valid_report())
    bad = valid_report()
    bad["findings"][0]["kind"] = "tumour"  # not in enum
    with pytest.raises(ValidationError):
        t.model.model_validate(bad)
    missing = valid_report()
    del missing["exam"]["slice_mm"]  # every key is required (explicit null)
    with pytest.raises(ValidationError):
        t.model.model_validate(missing)
    extra = valid_report()
    extra["exam"]["other"] = 1
    with pytest.raises(ValidationError):
        t.model.model_validate(extra)


def test_vllm_schema(typed_template_dir: Path):
    t = load_template(typed_template_dir)
    s = t.json_schema
    text = json.dumps(s)
    assert (
        "$ref" not in text and "$defs" not in text and "title" not in text and "default" not in text
    )
    assert check_schema_portability(s) == []
    assert s["required"] == ["exam", "findings", "impression"]
    finding = s["properties"]["findings"]["items"]
    assert finding["additionalProperties"] is False
    assert finding["required"] == list(finding["properties"])
    assert next(iter(finding["properties"])) == "evidence"  # YAML order kept
    side = finding["properties"]["side"]
    assert {"type": "null"} in side["anyOf"]
    assert finding["properties"]["kind"] == {
        "enum": ["nodule", "emphysema", "other"],
        "type": "string",
    }
    assert finding["properties"]["size"]["anyOf"][1] == {"type": "null"}


def test_portability_check_flags_unknown_keywords():
    assert check_schema_portability({"type": "string", "pattern": "x"}) == ["$: keyword 'pattern'"]


def test_reader_view(typed_template_dir: Path):
    t = load_template(typed_template_dir)
    view = t.reader_view
    assert '"non_contrast" | "contrast_enhanced" | null,  // from TECHNIQUE' in view
    assert '"kind": "nodule" | "emphysema" | "other",' in view
    assert "<number> | null" in view
    assert '"impression": "main conclusion"' in view
    assert "object or null" in view
    assert view in t.system_prompt  # default prompt includes it
    assert "not mentioned" in t.system_prompt


def test_hash_changes_with_prompt(typed_template_dir: Path):
    h1 = load_template(typed_template_dir).hash
    assert load_template(typed_template_dir).hash == h1
    (typed_template_dir / "system_prompt.md").write_text("Custom {{reader_view}}")
    assert load_template(typed_template_dir).hash != h1


def test_user_prompt_needs_report_placeholder(typed_template_dir: Path):
    (typed_template_dir / "user_prompt.md").write_text("no placeholder")
    with pytest.raises(TemplateError, match="report"):
        load_template(typed_template_dir)


def test_render_messages(typed_template_dir: Path):
    msgs = load_template(typed_template_dir).render_messages("REPORT TEXT")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "REPORT TEXT" in msgs[1]["content"] and "REPORT TEXT" not in msgs[0]["content"]


def test_field_catalog(typed_template_dir: Path):
    paths = {f["path"]: f for f in field_catalog(load_template(typed_template_dir).root)}
    assert paths["findings[].size.numbers[]"]["kind"] == "number"
    assert paths["exam.contrast"]["values"] == ["non_contrast", "contrast_enhanced"]


def test_snapshot_reloads_with_same_hash(typed_template_dir: Path, tmp_path: Path):
    t = load_template(typed_template_dir)
    t.snapshot(tmp_path / "snap")
    assert load_template(tmp_path / "snap").hash == t.hash


@pytest.mark.parametrize("name", ["chest_ct_findings", "chest_ct_simple"])
def test_repo_templates_load(name):
    t = load_template(name, REPO / "templates")
    assert t.config.feeder == "ct_rate"
    assert not t.warnings


def test_chest_ct_findings_example_validates():
    t = load_template("chest_ct_findings", REPO / "templates")
    report = {
        "exam": {
            "contrast": "non_contrast",
            "slice_thickness_mm": 1.5,
            "clinical_indication": None,
            "comparison_with_prior": None,
        },
        "findings": [
            {
                "evidence": "6x4 mm nodule",
                "finding_type": "lung_nodule",
                "name": "nodule",
                "status": "present",
                "region": "lung",
                "laterality": "left",
                "lobes": ["left_upper_lobe"],
                "sites": ["LUL_anterior"],
                "distribution": [],
                "count": "single",
                "count_number": None,
                "size": {
                    "numbers": [6, 4],
                    "unit": "mm",
                    "measure": "dimensions",
                    "qualifier": "exact",
                    "is_largest": False,
                },
                "severity": None,
                "change": None,
                "descriptors": [],
            }
        ],
        "negations": [
            {
                "evidence": "No enlarged lymph nodes",
                "finding_type": "lymphadenopathy",
                "kind": "explicit_negation",
                "qualifier": None,
                "regions": ["mediastinum"],
                "laterality": None,
                "sites": ["subcarinal"],
            }
        ],
        "impression_diagnoses": [],
        "limited_evaluation": [{"evidence": "could not be evaluated", "region": "mediastinum"}],
    }
    t.model.model_validate(report)


@pytest.mark.skipif(not RADFIELDS_TEMPLATES.is_dir(), reason="RadFields checkout not found")
def test_all_radfields_templates_load_unchanged():
    files = [
        p
        for p in RADFIELDS_TEMPLATES.rglob("*.yaml")
        if p.name not in {"template_index.yaml", "combined_exam_definitions.yaml"}
    ]
    assert len(files) > 100
    for path in files:
        t = load_template(path)
        assert t.json_schema["type"] == "object"
        # RadFields semantics: every leaf is a string field
        assert all(f["kind"] == "text" for f in field_catalog(t.root)), path


def test_render_reader_view_list_of_strings():
    view = render_reader_view(parse_instructions({"seq": ["T1", "T2"]}))
    assert '"seq": ["T1", ...]' in view


def test_schema_from_model_directly():
    model = build_model(parse_instructions({"a": {"_type": "boolean"}}))
    assert vllm_json_schema(model)["properties"]["a"] == {
        "anyOf": [{"type": "boolean"}, {"type": "null"}]
    }
