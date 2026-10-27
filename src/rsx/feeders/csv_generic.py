"""Generic CSV: one row = one report."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import pandas as pd

from .base import ReportFeeder, ReportUnit, join_sections, register_feeder


@register_feeder("csv")
class GenericCSVFeeder(ReportFeeder):
    """Any CSV with an id column and one or more text columns.

    Options (id_column and text_columns are required):
      id_column      column with a unique report id
      text_columns   list of columns concatenated into the report text
      headings       prefix each block with its column name (default true when several columns)
      sep            CSV separator (default ",")
    """

    default_options: ClassVar[dict[str, Any]] = {
        "id_column": None,
        "text_columns": None,
        "headings": None,
        "sep": ",",
    }

    def iter_units(self) -> Iterator[ReportUnit]:
        opts = self.options
        if not opts["id_column"] or not opts["text_columns"]:
            raise ValueError(
                "feeder 'csv' needs --feeder-opt id_column=<col> "
                "--feeder-opt text_columns=[<col>, ...]"
            )
        cols = opts["text_columns"]
        if isinstance(cols, str):
            cols = [cols]
        headings = len(cols) > 1 if opts["headings"] is None else bool(opts["headings"])
        df = pd.read_csv(self.input_path, dtype=str, keep_default_na=False, sep=opts["sep"])
        missing = [c for c in [opts["id_column"], *cols] if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.input_path}: missing column(s) {missing}; columns are {list(df.columns)}"
            )
        if df[opts["id_column"]].duplicated().any():
            raise ValueError(
                f"{self.input_path}: duplicated values in id column {opts['id_column']!r}"
            )
        self.member_column = opts["id_column"]
        sections = [(c, c.upper()) for c in cols]
        for row in df.to_dict("records"):
            text = join_sections(row, sections, headings=headings)
            if text:
                rid = str(row[opts["id_column"]])
                yield ReportUnit(key=rid, text=text, members=(rid,))
