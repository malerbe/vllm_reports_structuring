# Held-out phrasings

[← Q/A generation](README.md)

To measure whether the model understands a question or only recognizes its wording, part of the
phrasings of each template is never used in training.

## How phrasings are split

In `templates.yaml`:

```yaml
phrasing_holdout: {fraction: 0.2, seed: 0}
```

- Each phrasing goes to `heldout` when a hash of (seed, template id, text) falls below
  `fraction`. Adding or removing other phrasings does not move it.
- `{text: "...", split: heldout}` (or `split: train`) forces it.
- At least one phrasing per template stays in train, and (with `fraction > 0` and 2+ phrasings)
  at least one is held out: the one with the lowest hash. This guaranteed one can change when
  phrasings are added to a template that had none held out.
- The id of a phrasing (`qa.phrasing_id`) is a hash of (template id, text): editing a text gives
  a new id.

`gen.spec.phrasing_table()` lists every phrasing with its split:

```python
table = gen.spec.phrasing_table()
table.groupby(["template_id", "split"]).size().unstack(fill_value=0)
```

Concept display names (`concepts.yaml: display`) can be split the same way with
`display_holdout: {fraction: ..., seed: ...}` (0 by default: all names in train). When a concept
has no held-out name, the held-out mode uses all its names.

## Using the splits

`sampling_config.phrasings`:

| Value | Phrasings used |
|---|---|
| `train` (default) | train phrasings only |
| `heldout` | held-out phrasings only; templates without any are not drawn |
| `all` | every phrasing |

Evaluation protocol: the same validation images, two fixed sets.

```python
seen = valid.freeze(k=8, seed=0, sampling_config={"phrasings": "train"})
unseen = valid.freeze(k=8, seed=0, sampling_config={"phrasings": "heldout"})
```

The accuracy gap between `seen` and `unseen`, per template (`template_id` column), measures
generalization to unseen wordings. The current YAML has 3 to 5 phrasings per template, so
training sees only 1 to 3 wordings of each question (`score_severity`: 1); check with
`phrasing_table()` and add phrasings for a meaningful measure.

## Writing phrasings

- Use `{finding}` (and `{side}` for `presence_side`) and keep the meaning identical across the
  phrasings of a template: only the wording changes.
- Display names are singular without article, so that both "Does this CT show any {finding}?"
  and "How severe is the {finding}?" read well.
- More phrasings per template give a more reliable held-out measure. They can be drafted with an
  LLM and pasted in the YAML: the code only reads the YAML.
