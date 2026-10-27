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
