"""CT-RATE: one item per NIfTI volume."""

from __future__ import annotations

import re

from ..generator import QAGenerator, register_generator

_LEADING_ZEROS = re.compile(r"^(train|valid)_0*(\d)")
_SCAN_PREFIX = re.compile(r"^(train|valid)_\d+(_[a-z]+)?$")


@register_generator("ct_rate")
class CTRATEGenerator(QAGenerator):
    """Items are VolumeNames (<split>_<patient>_<scan>_<recon>.nii.gz). The reconstructions of
    a scan (_a_1, _a_2) are distinct items sharing one report, hence the same candidates.

    resolve() accepts a full VolumeName, the name without .nii.gz, or a prefix: "valid_192"
    (all volumes of patient 192), "valid_0192_a" (leading zeros are dropped).
    """

    member_column = "VolumeName"

    def normalize_query(self, query: str) -> str:
        q = _LEADING_ZEROS.sub(r"\1_\2", query.strip())
        if _SCAN_PREFIX.match(q):
            return q + "_"  # valid_192 must not match valid_1920_...
        if re.search(r"_\d+$", q) and not q.endswith(".nii.gz"):
            return q + ".nii.gz"
        return q
