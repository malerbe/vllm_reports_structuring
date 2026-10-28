# PyTorch dataset

[← Q/A generation](README.md)

`rsx.qa` does not depend on torch: the `Dataset` belongs to the training code. The example below
puts every expensive step in `__init__` and keeps `__getitem__` to one draw (~150 µs for 8
questions) plus the image loading. It was run as written (with a fake volume loader) on the
CT-RATE valid exports with `num_workers=2`.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rsx.qa import CTRATEGenerator

EXPORTS = "runs/chest_ct_findings/gpt-oss-120b/ctrate_{split}_v31/exports"


class CTRATEQADataset(Dataset):
    """One sample = one NIfTI volume + k questions drawn when it is loaded."""

    def __init__(self, split, k=8, sampling_config=None, stats=None, load_volume=None, seed=0):
        # every expensive step happens here, once: reading, facts, candidates, weights
        self.gen = CTRATEGenerator("chest_ct_findings", sampling_config=sampling_config)
        self.gen.connect_structured_report(EXPORTS.format(split=split), stats=stats)
        self.k, self.seed, self.epoch = k, seed, 0
        self.load_volume = load_volume

    def __len__(self):
        return len(self.gen)  # number of NIfTI volumes

    def set_epoch(self, epoch):
        self.epoch = epoch  # new questions every epoch, reproducible

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        qas = self.gen.sample_qa_from_item(idx, self.k, rng=rng)
        volume = str(self.gen.items[idx])
        return {
            "volume": volume,
            "image": self.load_volume(volume) if self.load_volume else None,
            "qa": qas,
        }


def collate_qa(batch):
    """Images stacked; questions kept as lists (an item may have fewer than k)."""
    return {
        "volume": [b["volume"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]) if batch[0]["image"] is not None else None,
        "question": [[q.question for q in b["qa"]] for b in batch],
        "options": [[q.options for q in b["qa"]] for b in batch],
        "answer": [torch.tensor([q.answer for q in b["qa"]], dtype=torch.long) for b in batch],
        "qtype": [[q.qtype for q in b["qa"]] for b in batch],
        "ordinal": [[q.ordinal for q in b["qa"]] for b in batch],
    }


train = CTRATEQADataset("train", load_volume=load_ct)                  # load_ct: your loader
valid = CTRATEQADataset("valid", stats=train.gen.stats, load_volume=load_ct)

# rare findings: draw the images with the rarity of their findings (see sampling.md)
sampler = WeightedRandomSampler(
    torch.as_tensor(train.gen.item_weights()), num_samples=len(train), replacement=True
)
loader = DataLoader(train, batch_size=16, sampler=sampler, num_workers=8, collate_fn=collate_qa)

for epoch in range(n_epochs):
    train.set_epoch(epoch)
    for batch in loader:
        ...
```

## Notes

- **Reproducibility.** The questions of an item depend only on (seed, epoch, index), whatever the
  worker that loads it. `set_epoch` must be called before the `DataLoader` iterator is created
  (the start of the `for` loop): workers copy the dataset at that point. With
  `persistent_workers=True` they keep their old copy, so derive the epoch differently (e.g. pass it
  through the sampler indices).
- **Memory.** The candidates are flat numpy arrays (~120 MB for CT-RATE train, `gen.describe()["candidates_MB"]`): forked workers share
  them without copying. The generator also keeps the intermediate tables (`gen.facts`, a few
  hundred MB) for inspection; they are never touched during sampling.
- **Fewer than k questions.** An item can return fewer than `k` questions (few candidates, or
  `answer_balance.drop_excess`); hence lists in the batch. `fill: rephrase` completes to k.
- **Quality flags.** Questions with alerts are drawn by default; see the
  [caution](sampling.md#quality-flags) to leave them out. `qa.flags` is available per question.
- **Losses.** `qtype` and `ordinal` let you pick the head / loss per question: binary for noul,
  categorical for choice, ordinal (options are ordered) for score.

## Fixed validation sets

Sampling new questions at every evaluation makes the curves noisy. Freeze them once:

```python
seen = valid.gen.freeze(k=8, seed=0)                                          # train phrasings
unseen = valid.gen.freeze(k=8, seed=0, sampling_config={"phrasings": "heldout"})
seen.to_parquet("runs/qa/valid_qa_seen.parquet")      # runs/ is git-ignored: these files
unseen.to_parquet("runs/qa/valid_qa_unseen.parquet")  # hold report sentences (evidence)
```

One row per question (`item`, `question`, `options`, `answer`, `qtype`, `ordinal`, `template_id`,
`phrasing_id`, `concept`, `flags`, `evidence`). A validation `Dataset` then reads these rows
grouped by `item` instead of calling the generator. The gap between `seen` and `unseen` measures
generalization to [unseen phrasings](phrasings_holdout.md).
