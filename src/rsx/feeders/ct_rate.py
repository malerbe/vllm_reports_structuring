"""CT-RATE reports CSV (train_reports.csv / validation_reports.csv)."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, ClassVar

import pandas as pd

from .base import ReportFeeder, ReportUnit, join_sections, register_feeder

log = logging.getLogger(__name__)


@register_feeder("ct_rate")
class CTRATEFeeder(ReportFeeder):
    """CT-RATE: one row per reconstructed volume
    (VolumeName = <split>_<patient>_<study>_<recon>.nii.gz).

    Reconstructions of the same scan share one report. With group_reconstructions (default) they
    form one unit whose members are all their VolumeNames, so each report is structured once.
    If the texts of a group differ, the group is split per distinct text (with a warning).

    Options:
      id_column               column holding the volume name (default VolumeName)
      sections                [[column, heading], ...] concatenated into the report text
      group_reconstructions   group <scan>_<n>.nii.gz rows into one unit (default true)
      group_regex             regex with a named group "key" giving the scan id
    """

    member_column = "VolumeName"
    default_options: ClassVar[dict[str, Any]] = {
        "id_column": "VolumeName",
        "sections": [
            ["ClinicalInformation_EN", "CLINICAL INFORMATION"],
            ["Technique_EN", "TECHNIQUE"],
            ["Findings_EN", "FINDINGS"],
            ["Impressions_EN", "IMPRESSION"],
        ],
        "group_reconstructions": True,
        "group_regex": r"^(?P<key>.+)_\d+\.nii\.gz$",
    }

    def iter_units(self) -> Iterator[ReportUnit]:
        opts = self.options
        df = pd.read_csv(self.input_path, dtype=str, keep_default_na=False)
        id_col = opts["id_column"]
        sections = [tuple(s) for s in opts["sections"]]
        missing = [c for c in [id_col, *(c for c, _ in sections)] if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.input_path}: missing column(s) {missing}; columns are {list(df.columns)}"
            )
        regex = re.compile(opts["group_regex"])

        groups: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
        for row in df.to_dict("records"):
            volume = row[id_col]
            m = regex.match(volume) if opts["group_reconstructions"] else None
            key = m.group("key") if m else volume
            groups.setdefault(key, []).append((volume, join_sections(row, sections)))

        n_split = 0
        for key, rows in groups.items():
            by_text: OrderedDict[str, list[str]] = OrderedDict()
            for volume, text in rows:
                by_text.setdefault(text, []).append(volume)
            if len(by_text) > 1:
                n_split += 1
                log.warning(
                    "scan %s: %d different report texts, structured separately", key, len(by_text)
                )
            for i, (text, volumes) in enumerate(by_text.items()):
                if not text:
                    log.warning("scan %s: empty report text, skipped (%s)", key, volumes)
                    continue
                yield ReportUnit(
                    key=key if len(by_text) == 1 else f"{key}#{i}",
                    text=text,
                    members=tuple(volumes),
                )
        log.info(
            "ct_rate: %d rows -> %d scans (%d with differing texts)", len(df), len(groups), n_split
        )
