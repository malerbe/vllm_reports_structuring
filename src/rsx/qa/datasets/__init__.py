"""One QAGenerator subclass per dataset. Add a dataset: create datasets/<name>.py with a
@register_generator class and import it below."""

from . import ct_rate  # noqa: F401  (registration)
from .ct_rate import CTRATEGenerator

__all__ = ["CTRATEGenerator"]
