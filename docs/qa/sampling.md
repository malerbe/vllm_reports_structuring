# Sampling configuration

[← Q/A generation](README.md)

The `sampling_config` decides which of an item's candidate questions are drawn, and how often.
It is a dict, validated (unknown keys raise an error), built in layers, later ones winning:

1. defaults of [`SamplingConfig`](../../src/rsx/qa/sampling.py)
2. [`qa_templates/<set>/sampling.yaml`](../../qa_templates/chest_ct_findings/sampling.yaml)
3. `sampling:` of the [dataset overlay](templates.md#dataset-overlays)
4. `CTRATEGenerator(..., sampling_config={...})` or `gen.set_sampling_config({...})`
5. the dict passed to one `sample_qa_from_item(item, k, {...})` call

Each config is compiled once into one weight per candidate (0.1–0.3 s on CT-RATE train), and the
last 8 are cached.

```yaml
qtype: {noul: 0.4, choice: 0.3, score: 0.3}
templates: {}                      # template_id -> multiplier, 0 disables
rarity: {alpha: 0.5, max_ratio: 50, scope: present}
answer_balance: {beta: 1.0, max_ratio: 20, min_count: 0, iterations: 10, k: 8, drop_excess: false}
which_present: {min_negated: 20, balance: 1.0}
flag_weights: {location_alert: 1, lat_lobe_conflict: 1, site_lobe_conflict: 1,
               subset_of_invalid: 1, size_warning: 1, closed_world: 1, derived: 1,
               low_evidence: 1}
low_evidence_threshold: 90
phrasings: train                   # train | heldout | all
shuffle_choice_options: true
fill: short                        # short | rephrase
item_weights: {mode: max}          # max | sum
```

## How k questions are drawn

For one item, `sample_qa_from_item(item, k)`:

1. draws how many questions of each type: multinomial(k, `qtype`), renormalized over the types
   the item has candidates for. If a type runs short, the rest goes to the other types;
2. within each type, draws that many distinct candidates with probability proportional to their
   weight (Gumbel top-k, no repeat);
3. picks a phrasing of the requested split and a display name, shuffles the options of choice
   questions (score options keep their order).

An item with fewer than k candidates gives fewer questions (`fill: short`); `fill: rephrase`
completes with already drawn candidates in another wording.

The weight of a candidate is the product of the factors below.

## Question types: `qtype` and `templates`

`qtype` gives the share of each type. On CT-RATE most items have only 1–2 score candidates, so
the share of score questions you get is lower than asked (measured on train with 0.4/0.3/0.3:
noul 0.55, choice 0.31, score 0.14). Raising `score` moves the balance as far as the data allow.

`templates` multiplies the weight of a template within its type (`{choice_lobe: 3}`); 0 turns it
off.

## Rare findings: `rarity` and `item_weights()`

A concept present in `n` reports gets the weight `(n_max / n) ** alpha` (at most `max_ratio`),
where `n_max` is the count of the most frequent concept (`lung_nodule`). With `scope: present`
this applies only to the candidates whose concept is present in the item (rare positives);
`scope: all` also upweights the No answers about rare concepts.

Inside one item, a weight can only choose among that item's questions: a pneumothorax is asked
about at most as often as images with a pneumothorax are loaded. To show rare findings more
often, also **draw the items** with `gen.item_weights()` (one weight per item: the rarity weight of
its rarest present concept, `mode: sum` for the sum), e.g. with a `WeightedRandomSampler`
([example](pytorch_dataset.md)).

Measured on CT-RATE train, questions about a present rare concept (pneumothorax, reticulation,
esophageal abnormality, pericardial thickening, lung mass, pericardial effusion):

| alpha | uniform items | items drawn with `item_weights()` |
|---|---|---|
| 0 | 2.8 % | 3.0 % |
| 0.5 | 3.1 % | 5.2 % |
| 1 | 3.2 % | 6.8 % |

## Answer balance

`answer_balance` makes the answers of each (template, concept) equally frequent, so that the
model cannot answer from the concept alone ('esophageal abnormality → No').

- `beta`: 1 = equal answers, 0 = off.
- Start: weight `(n_tc / (K * n_tca)) ** beta` from the counts of the reference stats.
- `iterations` rounds of calibration on the expected number of draws (`k` questions per item):
  a weight only acts within its item, so frequent answers must be pushed down for rare ones to
  catch up. Factors stay within `[1 / max_ratio, max_ratio]`.
- `min_count > 0`: a (template, concept) needs 2 answers with at least `min_count` candidates,
  else it is not drawn. Under the explicit-negation rule, many concepts are (almost) always Yes
  (ground-glass 7729 Yes / 19 No in train): `min_count: 20` drops their noul questions; they
  still appear in choice / score questions.
- `drop_excess: true`: an item whose only noul candidates are No still fills its noul quota with
  them, whatever their weight. With `drop_excess`, a drawn candidate is kept with probability
  (fewest expected draws among its (template, concept)'s answers / its own): answers become
  balanced, but items return fewer than k questions (7.4 instead of 8 on CT-RATE with its
  closed world, 6.4 without).

## Quality flags

Every candidate carries the flags of the report entries behind its answer:

| Flag | Meaning |
|---|---|
| `location_alert` | laterality / lobes / sites not supported by the evidence (postprocess check) |
| `lat_lobe_conflict` | laterality contradicts the lobes |
| `site_lobe_conflict` | a segment outside the listed lobes |
| `subset_of_invalid` | `subset_of` points to an invalid entry |
| `size_warning` | size with an unusable unit |
| `closed_world` | a No that comes from "not mentioned" (`concepts.closed_world` / `closed_world_if`) |
| `derived` | a Yes added by a `concepts.finding_rules` rule (crazy paving → septal thickening, ...) |
| `low_evidence` | `evidence_fuzzy` < `low_evidence_threshold` (quote not found in the report) |

`flag_weights` multiplies the weight of the flagged candidates: 1 keeps them as they are,
0.2 makes them rarer, **0 removes them**.

> [!CAUTION]
> **All flag weights are 1 by default: candidates with alerts or conflicts ARE sampled.**
> On CT-RATE train this concerns 22k candidates with `location_alert`, 6.5k `subset_of_invalid`,
> 1.3k `lat_lobe_conflict`, 0.9k `site_lobe_conflict`. To leave them out, set their weight to 0:
>
> ```python
> gen.set_sampling_config({"flag_weights": {
>     "location_alert": 0, "lat_lobe_conflict": 0, "site_lobe_conflict": 0,
>     "subset_of_invalid": 0, "size_warning": 0, "low_evidence": 0,
> }})
> ```
>
> or put the same `flag_weights` block in `sampling.yaml` / the dataset overlay.

> [!CAUTION]
> A flag marks a whole candidate: a `location_alert` nodule is also removed from the presence and
> severity questions, not only from the laterality ones. Use a weight between 0 and 1 to keep them
> but rarer.

> [!CAUTION]
> With the CT-RATE overlay, **most No answers come from the closed world** (461k of the 642k
> `noul_presence` No candidates on train): they rely on the assumption that a finding is reported
> when seen ([closed world](question_types.md#closed-world)). `flag_weights.closed_world` below 1
> makes them rarer; 0 brings back the explicit-negation-only behaviour, and also removes them
> from the distractors of `choice_which_present`.

> [!CAUTION]
> `derived` Yes answers (8k candidates on CT-RATE train) come from regex rules on the evidence,
> not from the extraction itself. Set `flag_weights.derived: 0` to leave them out.

`qa.flags` lists the flags of each drawn question, so they can also be filtered or logged in the
training loop.

## `choice_which_present`

Options = the present concept + 3 concepts negated in the same report. CT-RATE reports negate
almost always the same 10–15 concepts (esophagus, lymph nodes, effusions, ...), and some concepts
are never negated (ground-glass, atelectasis, emphysema): without care, the option names alone
give the answer.

- `min_negated`: a concept is an answer only if it is negated in ≥ `min_negated` reports (so it
  also shows up as a distractor).
- `balance`: answer and distractor weights are scaled by `((n_present + 1) / (n_negated + 1)) **
  (∓balance)` so that each concept is answer and distractor in similar proportions.

Blind accuracy (options only, chance 0.25) on CT-RATE train: 0.90 without these settings and
without closed world, 0.60 with these settings, **0.31** with the closed world of the CT-RATE
overlay (its No widen the distractor pools). Turn the template down
(`templates: {choice_which_present: 0.3}`) if this is too high for your use.

## Shortcut check

`python -m rsx qa blind <exports> [--weighted]` (or `rsx.qa.diagnostics.blind_baseline(gen)`)
samples questions, learns text-only priors on half of them (most frequent answer per
(template, concept); for `choice_which_present` the option most often right) and predicts the
other half. Accuracy well above chance = answerable without the image.

CT-RATE train, 2026-10-28, uniform items. "Explicit only": no closed world, no extraction fixes;
the other columns use the CT-RATE overlay (closed world) and the `concepts.yaml` rules:

| Template | Chance | Explicit only | Defaults | + `drop_excess` |
|---|---|---|---|---|
| noul_presence | 0.50 | 0.71 | 0.52 | 0.52 |
| noul_presence_side | 0.50 | 0.66 | 0.52 | 0.51 |
| choice_laterality | 0.33 | 0.53 | 0.50 | 0.52 |
| choice_lobe | 0.20 | 0.27 | 0.24 | 0.21 |
| choice_which_present | 0.25 | 0.60 | 0.31 | 0.31 |
| score_severity | 0.33 | 0.64 | 0.63 | 0.42 |
| score_count | 0.50 | 0.65 | 0.66 | 0.57 |
| score_nodule_size | 0.17 | 0.35 | 0.38 | 0.22 |
| score_effusion_thickness | 0.20 | 0.28 | 0.31 | 0.22 |
| score_aortic_diameter | 0.25 | 0.43 | 0.45 | 0.31 |

The closed world fixes the noul and which-present shortcuts. The score shortcuts come from the
per-item quota and only `drop_excess` reduces them (7.4 questions per item instead of 8, score
share 7% instead of 14%).

Run it again after changing templates, concepts or the sampling config.

## Valid / test generators: use the train stats

Rarity and answer balance start from counts. Pass the train counts to the other generators so
that "rare" means the same everywhere:

```python
train = CTRATEGenerator("chest_ct_findings").connect_structured_report(TRAIN_EXPORTS)
valid = CTRATEGenerator("chest_ct_findings").connect_structured_report(
    VALID_EXPORTS, stats=train.stats
)
```

`train.stats.to_dict()` / `QAStats.from_dict(...)` save and load them (JSON-compatible).

## Fixed evaluation sets

```python
frozen = valid.freeze(k=8, seed=0)                                    # every item
frozen.to_parquet("runs/qa/valid_qa_seen_phrasings.parquet")  # git-ignored folder
valid.freeze(k=8, seed=0, sampling_config={"phrasings": "heldout"}).to_parquet(
    "runs/qa/valid_qa_heldout_phrasings.parquet"
)
```

Each item's questions depend only on (seed, item index), so the set is the same at every
epoch and can be stored with the checkpoints.
