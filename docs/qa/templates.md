# Templates

[← Q/A generation](README.md)

A **template set** is a folder `qa_templates/<name>/`:

| File | Content |
|---|---|
| `templates.yaml` | the question templates |
| `concepts.yaml` | how findings and negations become concepts ([details](question_types.md)) |
| `sampling.yaml` | the default [sampling configuration](sampling.md) |
| `datasets/<dataset>.yaml` | optional [overlay](#dataset-overlays) for one dataset |

`chest_ct_findings` matches the exports of the `chest_ct_findings` structuring template.

## A template

```yaml
- id: choice_lobe                 # unique; also the key of sampling_config.templates
  qtype: choice                   # noul | choice | score
  builder: field                  # how candidates and answers are built (below)
  where: {region: [lung]}         # which candidates the template is valid for
  answer:                         # builder parameters
    field: lobe
    map: {lingula: left_upper_lobe}
    options:                      # value: label (in this order for a score template)
      right_upper_lobe: right upper lobe
      right_middle_lobe: right middle lobe
      right_lower_lobe: right lower lobe
      left_upper_lobe: left upper lobe
      left_lower_lobe: left lower lobe
  phrasings:
    - "In which lobe is the {finding} located?"
    - "Which lobe contains the {finding}?"
    - {text: "Which pulmonary lobe is affected by the {finding}?", split: heldout}
```

- `{finding}` is replaced by a display name of the concept; `presence_side` also fills `{side}`.
  A phrasing may leave out `{finding}`.
- `split: heldout` forces a phrasing into the held-out set; the others are split by hash
  ([held-out phrasings](phrasings_holdout.md)).
- Templates are checked when the generator is created (unknown builder, unknown placeholder,
  fewer than 2 options) and when it connects to the exports (unknown `where` column). Unknown
  ids in `sampling_config.templates` raise when the config is compiled.

## `where`

`where` filters the builder's source table. Every condition must hold:

| Condition | Meaning |
|---|---|
| `col: notnull` / `col: null` | value present / missing |
| `col: [a, b]` | value in the list |
| `col: {not_in: [a, b]}` | value not in the list (missing values pass) |
| `col: {min: x, max: y}` | numeric bounds, inclusive |
| `col: value` | equality |

Columns of the source tables:

| Builder | Source | Columns |
|---|---|---|
| `presence` | state per (report, concept) | `finding_type`, `answer`, `fuzzy` |
| `presence_side` | state per (report, concept, side) | the same + `side` |
| `field`, `bins` | attributes of present concepts | `finding_type`, `laterality`, `lobe`, `region`, `severity`, `change`, `size_mm_max`, `size_measure`, `count`, `n_entries`, `fuzzy` |
| `present_among_negated` | present concepts | `finding_type`, `fuzzy` |

## Builders

| Builder | Type | What it asks | `answer` parameters |
|---|---|---|---|
| `presence` | noul | is the concept present? | none |
| `presence_side` | noul | is it present on the right / left? | `sides: {right: ..., left: ...}` (labels) |
| `field` | choice, score | which value has an attribute? | `field`, `options` (value: label), optional `map` (value renaming before matching) |
| `bins` | score | which interval holds a number? | `field`, `edges` (sorted), `labels` (one more than edges); bin i = edges[i-1] ≤ x < edges[i] |
| `present_among_negated` | choice | which of n options is present? | `n_options` (template key, default 4) |

A `field` value that is not among the `options` gives no question. `map` merges values: CT-RATE
maps `severe` to `moderate` and labels it "moderate or severe".

### Adding a builder

Write a function `(facts, template) -> DataFrame` in [`builders.py`](../../src/rsx/qa/builders.py)
(or any module imported before the generator is created) and register it:

```python
from rsx.qa.builders import candidate_rows, option_values, register_builder
from rsx.qa.templates import apply_where


def _options(tpl, noul_options):
    return option_values(tpl)[1]


@register_builder("distribution", options=_options)
def distribution(facts, tpl):
    """Choice: central or peripheral?"""
    src = apply_where(facts.agg, tpl.where, tpl.id)
    ...                                     # compute one answer index per row
    return candidate_rows(src, answer, present=True)
```

It must return the columns `unit_idx, concept_idx, slot, answer, flags, fuzzy, evidence, present`
(`candidate_rows` builds them). `answer` is an index into the options, or -1 when the options are drawn at
sampling time (then `options=` returns None and the generator needs a matching branch, like
`present_among_negated`). If a new attribute is needed, add it to `facts.agg` in
[`facts.py`](../../src/rsx/qa/facts.py).

## `concepts.yaml`

How findings and negations become concepts and Yes / No answers
([details](question_types.md)):

| Key | Content |
|---|---|
| `other_min_count` | an `other` finding becomes the concept `other:<name>` if the name occurs in this many reports |
| `negation_kinds` | negation kinds that give a No (`explicit_negation`, `normal_statement`) |
| `neutral_qualifiers` | per type (or `_all`), qualifiers that do not narrow a negation (regex, full match) |
| `implies` | child → parent types: a present child blocks the No of its parents |
| `closed_world` | types for which "not mentioned" = No (set per dataset; a type may not be listed twice nor also in `closed_world_if`) |
| `closed_world_if` | `{type: {regions: [...], evidence: regex}}`: closed world only in reports that mention the region (set per dataset) |
| `exclude` | concepts never asked about (names or glob patterns: `"other:*"`) |
| `negation_rules` | `{finding_type, evidence, evidence_not, action: drop \| retype, to}`: fix mistyped negations |
| `finding_rules` | `{finding_type, evidence, evidence_not, max_short_axis_mm, action: add \| negate, to}`: add a missed finding (flag `derived`), or turn a finding into a No of another type |
| `display`, `display_holdout` | display names of each type, and their held-out fraction |

Rules are applied first, in order, with a case-insensitive regex search on the evidence:

```yaml
negation_rules:
  - {finding_type: arterial_wall_calcification, evidence_not: "calcif|atherom|plaque", action: drop}
  - {finding_type: lymph_node_not_enlarged, action: retype, to: lymphadenopathy}
finding_rules:
  - {finding_type: ground_glass_opacity, evidence: "crazy[- ]?paving", action: add, to: interlobular_septal_thickening}
```

A finding added by a rule keeps the location of its source (laterality, lobes, region) but not its
severity, size, count or change.

## Dataset overlays

The same template set may serve several datasets. Changes for one dataset go in
`qa_templates/<set>/datasets/<generator name>.yaml`, which the generator of that name loads
automatically (`CTRATEGenerator` loads `datasets/ct_rate.yaml`):

```yaml
concepts:            # each key replaces the concepts.yaml key
  closed_world: [ground_glass_opacity, emphysema]
  closed_world_if: {liver_lesion: {regions: [liver]}}
templates:           # per template id: each key replaces the template's key; null drops it
  score_severity:
    answer:
      field: severity
      map: {severe: moderate}
      options: {minimal: minimal, mild: mild, moderate: "moderate or severe"}
  score_aortic_diameter: null
sampling:            # merged into sampling.yaml
  qtype: {noul: 0.3, choice: 0.35, score: 0.35}
```

`CTRATEGenerator(..., overlay="other.yaml")` uses another file; `overlay=False` uses none. The
generic `QAGenerator` uses no overlay. `gen.describe()["overlay"]` shows which one is active.
