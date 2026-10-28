# Adding a dataset

[← Q/A generation](README.md)

`QAGenerator` holds everything that does not depend on the dataset. Each dataset gets a subclass
in [`src/rsx/qa/datasets/`](../../src/rsx/qa/datasets), registered under a name:

```python
# src/rsx/qa/datasets/my_dataset.py
from ..generator import QAGenerator, register_generator


@register_generator("my_dataset")
class MyDatasetGenerator(QAGenerator):
    """Items are the image ids of the `image_id` column of the exports."""

    member_column = "image_id"              # column of the exports naming an item

    def item_table(self, store):
        # optional: which items to train on (default: every member of the exports)
        items = super().item_table(store)
        return items[~items["item"].str.endswith("_localizer")]

    def normalize_query(self, query):
        # optional: how resolve() reads a user query (default: strip)
        return query.strip().lower()
```

Then import it in [`datasets/__init__.py`](../../src/rsx/qa/datasets/__init__.py):

```python
from . import ct_rate, my_dataset  # noqa: F401  (registration)
```

The exports come from `rsx export` of a run whose feeder wrote `member_column`
(`ReportFeeder.member_column`). `CTRATEGenerator` ([`ct_rate.py`](../../src/rsx/qa/datasets/ct_rate.py))
is a complete example: items are `VolumeName`s, and `normalize_query` accepts `valid_0192`.

## Dataset-specific choices

Put them in an overlay `qa_templates/<template set>/datasets/my_dataset.yaml` rather than in the
shared YAML files: which types follow the closed world, merged answer levels, dropped templates,
sampling defaults. See [dataset overlays](templates.md#dataset-overlays).

## A new structuring template

A template set reads the columns of one structuring template's exports (`finding_type`, `kind`,
`laterality`, `lobes`, `severity`, `size_mm_max`, ...). For another structuring template, create
`qa_templates/<name>/` with its own `concepts.yaml` and `templates.yaml`; the builders work
as long as the exports have the `findings` / `negations` / `reports` tables with these columns.

## Checks

```bash
python -m rsx qa stats <exports> --dataset my_dataset
python -m rsx qa preview <exports> --dataset my_dataset --evidence
python -m rsx qa blind <exports> --dataset my_dataset
```
