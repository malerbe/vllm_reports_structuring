"""Dataset readers. Add a dataset: create feeders/<name>.py with a @register_feeder class and
import it below."""

from . import csv_generic, ct_rate  # noqa: F401  (registration)
from .base import (
    ReportFeeder,
    ReportUnit,
    available_feeders,
    get_feeder,
    make_feeder,
    parse_feeder_options,
    register_feeder,
)

__all__ = [
    "ReportFeeder",
    "ReportUnit",
    "available_feeders",
    "get_feeder",
    "make_feeder",
    "parse_feeder_options",
    "register_feeder",
]
