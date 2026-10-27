"""Check that evidence quotes really come from the report (the JSON schema cannot enforce it)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rapidfuzz import fuzz

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = dict.fromkeys(map(ord, "‘’‚‛“”„‟´`"), "'")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Casefold, NFKC, unify dashes/quotes, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES).casefold()
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def verify(span: Any, source: str) -> dict[str, Any]:
    """verbatim: exact substring; normalized: substring after normalize(); fuzzy: best partial
    match score 0-100 (rapidfuzz partial_ratio). Empty or non-string spans give all False / 0."""
    if not isinstance(span, str) or not span.strip():
        return {"verbatim": False, "normalized": False, "fuzzy": 0.0}
    verbatim = span.strip() in source
    n_span, n_src = normalize(span), normalize(source)
    normalized = verbatim or (bool(n_span) and n_span in n_src)
    score = 100.0 if normalized else round(fuzz.partial_ratio(n_span, n_src), 1)
    return {"verbatim": verbatim, "normalized": normalized, "fuzzy": score}
