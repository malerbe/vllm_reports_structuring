"""Question/answer generation from structured reports (rsx export tables).

    from rsx.qa import CTRATEGenerator
    gen = CTRATEGenerator("chest_ct_findings")
    gen.connect_structured_report("runs/chest_ct_findings/gpt-oss-120b/ctrate_train_v31/exports")

Documentation: docs/qa/README.md.
"""

from .builders import BUILDERS, register_builder
from .datasets import CTRATEGenerator
from .facts import FLAGS
from .generator import (
    QA,
    QAGenerator,
    QAStats,
    available_generators,
    get_generator,
    register_generator,
)
from .sampling import SamplingConfig
from .store import StructuredReportStore
from .templates import QTYPES, load_qa_spec

__all__ = [
    "BUILDERS",
    "FLAGS",
    "QA",
    "QTYPES",
    "CTRATEGenerator",
    "QAGenerator",
    "QAStats",
    "SamplingConfig",
    "StructuredReportStore",
    "available_generators",
    "get_generator",
    "load_qa_spec",
    "register_builder",
    "register_generator",
]
