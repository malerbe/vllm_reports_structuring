from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RADFIELDS_TEMPLATES = REPO.parent / "labels_extraction" / "RadFields" / "templates" / "de"


@pytest.fixture
def ct_rate_csv(tmp_path: Path) -> Path:
    """Mini CT-RATE file: scan a has 2 reconstructions (same text), b one, c two with different
    texts, d has the same text as a (another patient)."""
    a = (
        "Not given.",
        "Non-contrast.",
        "A 6x4 mm nodule in the right upper lobe. No pleural effusion.",
        "Nodule.",
    )
    rows = [
        ("valid_1_a_1.nii.gz", *a),
        ("valid_1_a_2.nii.gz", *a),
        ("valid_2_a_1.nii.gz", "Cough.", "Contrast-enhanced.", "Mild emphysema.", "Emphysema."),
        ("valid_3_a_1.nii.gz", "", "Non-contrast.", "Text one.", ""),
        ("valid_3_a_2.nii.gz", "", "Non-contrast.", "Text two.", ""),
        ("valid_4_b_1.nii.gz", *a),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "VolumeName",
            "ClinicalInformation_EN",
            "Technique_EN",
            "Findings_EN",
            "Impressions_EN",
        ],
    )
    path = tmp_path / "reports.csv"
    df.to_csv(path, index=False)
    return path


TYPED_TEMPLATE = textwrap.dedent(
    """
    _vocab:
      sides: &sides [left, right]
    exam:
      contrast: {_type: enum, values: [non_contrast, contrast_enhanced], _hint: "from TECHNIQUE"}
      slice_mm: {_type: number}
    findings:
      - evidence: "verbatim quote"
        kind: {_type: enum, values: [nodule, emphysema, other], _nullable: false}
        side: {_type: enum, values: *sides}
        count: {_type: integer}
        largest: {_type: boolean}
        size:
          _nullable: true
          numbers:
            - {_type: number}
          unit: {_type: enum, values: [mm, cm], _nullable: false}
    impression: "main conclusion"
    """
)


@pytest.fixture
def typed_template_dir(tmp_path: Path) -> Path:
    d = tmp_path / "typed"
    d.mkdir()
    (d / "template.yaml").write_text(TYPED_TEMPLATE)
    (d / "config.yaml").write_text(
        "feeder: ct_rate\nmissing_value: not mentioned\nexport_tables: {findings: findings}\n"
    )
    return d


def _finding(unit: str, idx: int, finding_type: str, **kw) -> dict:
    row = {
        "unit_id": unit,
        "item_idx": idx,
        "evidence": f"{finding_type} evidence {idx}",
        "finding_type": finding_type,
        "name": finding_type.replace("_", " "),
        "subset_of": None,
        "status": "present",
        "region": "lung",
        "laterality": None,
        "lobes": "[]",
        "sites": "[]",
        "distribution": "[]",
        "descriptors": "[]",
        "count": None,
        "count_number": None,
        "severity": None,
        "change": None,
        "evidence_fuzzy": 100.0,
        "size_mm_max": None,
        "size.measure": None,
        "size.qualifier": None,
        "size_warning": None,
        "location_alert": False,
        "lat_lobe_conflict": False,
        "site_lobe_conflict": False,
        "subset_of_invalid": False,
    }
    row.update(kw)
    return row


def _negation(unit: str, idx: int, finding_type: str, **kw) -> dict:
    row = {
        "unit_id": unit,
        "item_idx": idx,
        "evidence": f"no {finding_type} {idx}",
        "finding_type": finding_type,
        "kind": "explicit_negation",
        "qualifier": None,
        "regions": "[]",
        "laterality": None,
        "sites": "[]",
        "evidence_fuzzy": 100.0,
    }
    row.update(kw)
    return row


QA_MEMBERS = {
    "u1": ["valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz"],
    "u2": ["valid_2_a_1.nii.gz"],
    "u10": ["valid_10_a_1.nii.gz"],
    "ufail": ["valid_3_a_1.nii.gz"],
}


@pytest.fixture
def qa_exports(tmp_path: Path) -> Path:
    """Mini `rsx export` folder (rows repeated per member like the real exports).

    u1: nodules right RUL (location_alert) + left LLL; mild emphysema without side; ground-glass;
        no pleural effusion; no lymph node 'of pathological size' (neutral qualifier);
        no 'suspicious' mass (restrictive qualifier); no infiltration (blocked: ground-glass);
        no pneumothorax on the right; heart size normal (normal_statement)
        no infection (an excluded concept)
    u2: pleural effusion left minimal 15 mm + right moderate; 'no pleural effusion' (conflict);
        no pneumothorax; a resolved consolidation (not present); crazy paving (rule: adds septal
        thickening); millimetric lymph nodes (rule: No lymphadenopathy); 'vascular structures
        normal' typed arterial_wall_calcification (rule: dropped); no liver lesion + an `other`
        liver cyst
    u10: a single 6 mm nodule (exactly on a bin edge); an uncertain lung mass; 'no pathological
        lymph node' typed lymph_node_not_enlarged (rule: retyped); no liver lesion
    ufail: status failed (dropped)
    """
    findings = [
        _finding(
            "u1",
            0,
            "lung_nodule",
            laterality="right",
            lobes='["right_upper_lobe"]',
            location_alert=True,
            count="single",
            size_mm_max=5.0,
            evidence_fuzzy=80.0,
        ),
        _finding("u1", 1, "lung_nodule", laterality="left", lobes='["left_lower_lobe"]'),
        _finding("u1", 2, "emphysema", severity="mild"),
        _finding("u1", 3, "ground_glass_opacity", laterality="bilateral"),
        _finding(
            "u2",
            0,
            "pleural_effusion",
            region="pleura",
            laterality="left",
            severity="minimal",
            size_mm_max=15.0,
            **{"size.measure": "thickness"},
        ),
        _finding(
            "u2", 1, "pleural_effusion", region="pleura", laterality="right", severity="moderate"
        ),
        _finding("u2", 2, "consolidation", change="resolved"),
        _finding(
            "u10",
            0,
            "lung_nodule",
            laterality="left",
            lobes='["lingula"]',
            count="single",
            size_mm_max=6.0,
        ),
        _finding("u10", 1, "lung_mass", status="uncertain"),
        _finding(
            "u2",
            3,
            "ground_glass_opacity",
            laterality="bilateral",
            severity="mild",
            evidence="Crazy paving pattern in both lungs.",
        ),
        _finding(
            "u2",
            4,
            "lymph_node_not_enlarged",
            region="mediastinum",
            evidence="Millimetric lymph nodes in the mediastinum.",
        ),
        _finding("u2", 5, "other", name="hepatic cyst", region="liver", evidence="Liver cyst."),
    ]
    negations = [
        _negation("u1", 0, "pleural_effusion"),
        _negation("u1", 1, "lymphadenopathy", qualifier="pathological size"),
        _negation("u1", 2, "lung_mass", qualifier="suspicious"),
        _negation("u1", 3, "lung_opacity"),
        _negation("u1", 4, "pneumothorax", laterality="right"),
        _negation("u1", 5, "cardiomegaly", kind="normal_statement"),
        _negation("u1", 6, "infection", evidence="No sign of infection."),
        _negation("u2", 0, "pleural_effusion", laterality="bilateral"),
        _negation("u2", 1, "pneumothorax"),
        _negation(
            "u2",
            2,
            "arterial_wall_calcification",
            evidence="Mediastinal main vascular structures are normal.",
        ),
        _negation("u2", 3, "liver_lesion", regions='["liver"]', evidence="No lesion in the liver."),
        _negation("u10", 0, "lymph_node_not_enlarged", evidence="No pathological lymph node."),
        _negation(
            "u10", 1, "liver_lesion", regions='["liver"]', evidence="No lesion in the liver."
        ),
    ]
    reports = [
        {
            "VolumeName": v,
            "unit_id": u,
            "status": "failed" if u == "ufail" else "ok",
            "exam.contrast": "non_contrast",
        }
        for u, vols in QA_MEMBERS.items()
        for v in vols
    ]

    def per_member(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"VolumeName": v, **r} for r in rows for v in QA_MEMBERS[r["unit_id"]]]
        )

    d = tmp_path / "exports"
    d.mkdir()
    pd.DataFrame(reports).to_parquet(d / "reports.parquet")
    per_member(findings).to_parquet(d / "findings.parquet")
    per_member(negations).to_csv(d / "negations.csv", index=False)  # CSV fallback
    return d


def valid_report() -> dict:
    return {
        "exam": {"contrast": "non_contrast", "slice_mm": None},
        "findings": [
            {
                "evidence": "A 6x4 mm nodule in the right upper lobe",
                "kind": "nodule",
                "side": "right",
                "count": None,
                "largest": None,
                "size": {"numbers": [6, 4], "unit": "mm"},
            },
            {
                "evidence": "invented quote",
                "kind": "other",
                "side": None,
                "count": 1,
                "largest": False,
                "size": None,
            },
        ],
        "impression": "Nodule.",
    }
