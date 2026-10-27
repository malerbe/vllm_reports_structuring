"""Work units: feeder units -> deduplicated, stably identified requests; sharding and selection."""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from .feeders import ReportUnit


@dataclass(frozen=True)
class WorkUnit:
    """One LLM request. id = sha1 of the text (stable across runs and input order)."""

    id: str
    text: str
    members: tuple[str, ...]
    keys: tuple[str, ...]


def text_id(text: str, salt: str = "") -> str:
    return hashlib.sha1((salt + "\0" + text if salt else text).encode()).hexdigest()[:16]


def build_work_units(units: Iterable[ReportUnit], text_dedup: bool = True) -> list[WorkUnit]:
    """Merge units with identical text (text_dedup) so each distinct report is sent once."""
    merged: OrderedDict[str, tuple[str, list[str], list[str]]] = OrderedDict()
    for u in units:
        uid = text_id(u.text) if text_dedup else text_id(u.text, salt=u.key)
        if uid in merged:
            _, members, keys = merged[uid]
            members.extend(m for m in u.members if m not in members)
            keys.append(u.key)
        else:
            merged[uid] = (u.text, list(u.members), [u.key])
    return [
        WorkUnit(id=i, text=t, members=tuple(m), keys=tuple(k)) for i, (t, m, k) in merged.items()
    ]


def shard_units(units: list[WorkUnit], num_shards: int, shard_id: int) -> list[WorkUnit]:
    """Deterministic split by id (independent of input order)."""
    if num_shards <= 1:
        return units
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id {shard_id} not in [0, {num_shards})")
    return [u for u in units if int(u.id, 16) % num_shards == shard_id]


def select_units(
    units: list[WorkUnit],
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    ids: Iterable[str] | None = None,
) -> list[WorkUnit]:
    """Subset: explicit ids (unit ids, keys or members), a random sample, or the first N."""
    out = units
    if ids is not None:
        wanted = set(ids)
        out = [u for u in out if u.id in wanted or wanted & set(u.keys) or wanted & set(u.members)]
    if sample is not None and sample < len(out):
        out = random.Random(seed).sample(sorted(out, key=lambda u: u.id), sample)
    if limit is not None:
        out = out[:limit]
    return out


def longest_first(units: list[WorkUnit]) -> list[WorkUnit]:
    """Send the longest reports first: shortens the tail of the run."""
    return sorted(units, key=lambda u: len(u.text), reverse=True)
