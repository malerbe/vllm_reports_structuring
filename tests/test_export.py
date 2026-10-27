from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from conftest import REPO, valid_report

from rsx.evidence import normalize, verify
from rsx.export import export_run, flatten
from rsx.results import Record, ResultsWriter
from rsx.template import load_template


def test_normalize_and_verify():
    src = "A 6x4 mm nodule in the right upper lobe. Pleural effusion–thickening was not detected."
    assert normalize("Pleural  Effusion-thickening.") == "pleural effusion thickening"
    assert verify("6x4 mm nodule", src) == {"verbatim": True, "normalized": True, "fuzzy": 100.0}
    v = verify("pleural effusion-thickening was not detected", src)
    assert not v["verbatim"] and v["normalized"]
    v = verify("nodule in the left lower lobe", src)
    assert not v["normalized"] and 50 < v["fuzzy"] < 100
    assert verify(None, src)["fuzzy"] == 0.0


def test_flatten():
    assert flatten({"a": {"b": 1, "c": None}, "d": [1, 2]}) == {
        "a.b": 1,
        "a.c": None,
        "d": "[1, 2]",
    }


def make_run(tmp_path: Path, template_dir: Path) -> Path:
    t = load_template(template_dir)
    run = tmp_path / "run"
    t.snapshot(run / "template")
    (run / "run.yaml").write_text(yaml.safe_dump({"member_column": "VolumeName"}))
    text = "FINDINGS:\nA 6x4 mm nodule in the right upper lobe."
    with ResultsWriter(run / "results.jsonl") as w:
        w.write(
            Record(
                id="u1",
                keys=["valid_1_a"],
                members=["valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz"],
                text=text,
                status="ok",
                parsed=valid_report(),
                template_hash=t.hash,
            )
        )
        w.write(
            Record(
                id="u2",
                keys=["valid_2_a"],
                members=["valid_2_a_1.nii.gz"],
                text="x",
                status="truncated",
                raw='{"exam":',
                error="finish_reason=length",
                template_hash=t.hash,
            )
        )
    return run


def test_export_run(tmp_path: Path, typed_template_dir: Path):
    run = make_run(tmp_path, typed_template_dir)
    written = export_run(run)
    reports = pd.read_parquet(written["reports.parquet"])
    assert list(reports["VolumeName"]) == [
        "valid_1_a_1.nii.gz",
        "valid_1_a_2.nii.gz",
        "valid_2_a_1.nii.gz",
    ]
    assert reports.loc[0, "exam.contrast"] == "non_contrast"
    assert pd.isna(reports.loc[0, "exam.slice_mm"])  # null stays NA, never False / ""
    assert reports.loc[0, "n_findings"] == 2
    findings = pd.read_parquet(written["findings.parquet"])
    assert len(findings) == 4  # 2 findings x 2 members
    first = findings[(findings.VolumeName == "valid_1_a_1.nii.gz") & (findings.item_idx == 0)].iloc[
        0
    ]
    assert first["evidence_verbatim"] and first["kind"] == "nodule" and first["size.unit"] == "mm"
    assert json.loads(first["size.numbers"]) == [6, 4]
    second = findings[findings.item_idx == 1].iloc[0]
    assert not second["evidence_normalized"]
    failures = pd.read_csv(written["failures.csv"])
    assert list(failures["status"]) == ["truncated"]
    summary = json.loads(written["summary.json"].read_text())
    assert summary["status"] == {"ok": 1, "truncated": 1}
    assert summary["evidence"]["findings"]["evidence"]["verbatim"] == 0.5


def test_chest_postprocess_size_mm():
    t = load_template("chest_ct_findings", REPO / "templates")
    from rsx.export import load_postprocess

    post = load_postprocess(t)
    df = pd.DataFrame(
        {"size.numbers": ["[6, 4]", "[1.2]", "[5]", None], "size.unit": ["mm", "cm", "none", None]}
    )
    out = post("findings", df)
    assert list(out["size_mm_max"][:2]) == [6.0, 12.0]
    assert out["size_mm_min"][0] == 4.0
    assert pd.isna(out["size_mm_max"][2]) and "unusable" in out["size_warning"][2]
    assert pd.isna(out["size_mm_max"][3])
    assert post("negations", df) is df
