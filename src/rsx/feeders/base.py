"""Feeder contract: turn one input dataset into report units (text to structure + member ids)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml


@dataclass(frozen=True)
class ReportUnit:
    """One report to structure.

    key      feeder-level identifier of the report (e.g. a CT-RATE scan "valid_1_a")
    text     report text sent to the LLM
    members  ids the result applies to (e.g. both reconstructions valid_1_a_1/_2.nii.gz);
             exports write one row per member, under the feeder's member_column name
    """

    key: str
    text: str
    members: tuple[str, ...]


class ReportFeeder(ABC):
    """Base class of dataset readers. Subclasses are registered with @register_feeder("name")."""

    name: ClassVar[str] = ""
    member_column: ClassVar[str] = "id"  # column name of member ids in the exported tables
    default_options: ClassVar[dict[str, Any]] = {}

    def __init__(self, input_path: str | Path, **options: Any):
        self.input_path = Path(input_path)
        unknown = set(options) - set(self.default_options)
        if unknown:
            raise ValueError(
                f"feeder {self.name!r}: unknown option(s) {sorted(unknown)}; "
                f"known: {sorted(self.default_options)}"
            )
        self.options = {**self.default_options, **options}

    @abstractmethod
    def iter_units(self) -> Iterator[ReportUnit]:
        """Yield the report units of the input file."""

    def describe(self) -> dict[str, Any]:
        """Stored in run.yaml."""
        return {"feeder": self.name, "input": str(self.input_path), "options": self.options}


_FEEDERS: dict[str, type[ReportFeeder]] = {}


def register_feeder(name: str):
    def deco(cls: type[ReportFeeder]) -> type[ReportFeeder]:
        if name in _FEEDERS:
            raise ValueError(f"feeder {name!r} registered twice")
        cls.name = name
        _FEEDERS[name] = cls
        return cls

    return deco


def available_feeders() -> dict[str, type[ReportFeeder]]:
    return dict(_FEEDERS)


def get_feeder(name: str) -> type[ReportFeeder]:
    if name not in _FEEDERS:
        raise KeyError(f"unknown feeder {name!r}; available: {sorted(_FEEDERS)}")
    return _FEEDERS[name]


def parse_feeder_options(items: list[str] | None) -> dict[str, Any]:
    """--feeder-opt key=value (value parsed as YAML: numbers, booleans, lists)."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--feeder-opt expects key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def make_feeder(
    name: str, input_path: str | Path, options: dict[str, Any] | None = None
) -> ReportFeeder:
    return get_feeder(name)(input_path, **(options or {}))


def join_sections(
    row: dict[str, Any], sections: list[list[str]] | list[tuple[str, str]], headings: bool = True
) -> str:
    """Concatenate columns as 'HEADING:\\n<text>' blocks, skipping empty values."""
    blocks = []
    for column, heading in sections:
        value = row.get(column)
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.lower() == "nan":
            continue
        blocks.append(f"{heading}:\n{value}" if headings and heading else value)
    return "\n\n".join(blocks)
