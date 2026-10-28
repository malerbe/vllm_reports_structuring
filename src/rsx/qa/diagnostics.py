"""Shortcut check: how well can the sampled questions be answered without the image?

blind_baseline() samples Q/A, learns text-only priors on one half and predicts the other half:
  fixed options (noul, choice, score)  the most frequent answer of (template, concept)
  drawn options (choice_which_present) the option whose text is most often the answer
An accuracy well above chance means the model can learn the answer from the question alone.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .generator import QA, QAGenerator


def blind_baseline(
    gen: QAGenerator,
    n_items: int = 20000,
    k: int = 8,
    sampling_config: dict[str, Any] | None = None,
    seed: int = 0,
    item_weights: bool = False,
) -> pd.DataFrame:
    """One row per template: questions, chance level and blind accuracy on the held-out half.
    item_weights=True draws the items like a WeightedRandomSampler on gen.item_weights()."""
    rng = np.random.default_rng(seed)
    if item_weights:
        p = gen.item_weights(sampling_config)
        items = rng.choice(len(gen), n_items, p=p / p.sum())
    else:
        items = rng.integers(len(gen), size=n_items)
    qas: list[QA] = [
        q for i in items for q in gen.sample_qa_from_item(int(i), k, sampling_config, rng)
    ]
    order = rng.permutation(len(qas))
    fit, test = [qas[i] for i in order[: len(qas) // 2]], [qas[i] for i in order[len(qas) // 2 :]]

    majority: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    shown, answered = collections.Counter(), collections.Counter()
    for q in fit:
        majority[q.template_id, q.concept][q.answer_text] += 1
        shown.update(q.options)
        answered[q.answer_text] += 1

    drawn = {i for i, opts in enumerate(gen.options) if opts is None}
    drawn_ids = {gen.template_ids[i] for i in drawn}
    rows: dict[str, list[float]] = collections.defaultdict(list)
    chance: dict[str, list[float]] = collections.defaultdict(list)
    for q in test:
        if q.template_id in drawn_ids:
            score = [answered[o] / shown[o] if shown[o] else 0.0 for o in q.options]
            guess = q.options[int(np.argmax(score))]
        else:
            counts = majority.get((q.template_id, q.concept))
            guess = counts.most_common(1)[0][0] if counts else q.options[0]
        rows[q.template_id].append(guess == q.answer_text)
        chance[q.template_id].append(1 / len(q.options))
    out = pd.DataFrame(
        [
            {
                "template_id": t,
                "questions": len(v),
                "chance": round(float(np.mean(chance[t])), 3),
                "blind_accuracy": round(float(np.mean(v)), 3),
            }
            for t, v in rows.items()
        ]
    )
    order = {t: i for i, t in enumerate(gen.template_ids)}
    return out.sort_values("template_id", key=lambda s: s.map(order)).reset_index(drop=True)
