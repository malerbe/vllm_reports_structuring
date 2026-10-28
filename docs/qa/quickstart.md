# Quick start

[← Q/A generation](README.md)

## From Python

```python
import numpy as np
from rsx.qa import CTRATEGenerator

gen = CTRATEGenerator("chest_ct_findings")           # qa_templates/chest_ct_findings + CT-RATE overlay
gen.connect_structured_report("runs/chest_ct_findings/gpt-oss-120b/ctrate_train_v31/exports")

qas = gen.sample_qa_from_item("train_10001_a_1.nii.gz", 8, rng=np.random.default_rng(0))
for qa in qas:
    print(qa.qtype, qa.question, qa.options, "->", qa.answer_text, qa.flags)
```

`connect_structured_report` takes about 10 s on the CT-RATE train exports (47k volumes); the
valid exports load in under a second.

### Items

An **item** is one NIfTI volume (`VolumeName`). The two reconstructions of a scan
(`_a_1`, `_a_2`) are two items that share one report, so they get the same candidate questions.

```python
len(gen)                          # 47149 items
gen.items[:2]                     # numpy array of VolumeNames
gen.resolve("train_010001")       # all volumes of patient 10001 (leading zeros are dropped)
gen.resolve("train_10001_a")      # ['train_10001_a_1.nii.gz', 'train_10001_a_2.nii.gz']
gen.sample_qa_from_item("train_10001_a", 8)   # a prefix works if its volumes share one report
gen.sample_qa_from_item(123, 8)               # or the item index (fastest, use it in a Dataset)
```

### One `QA`

| Field | |
|---|---|
| `question` | the rendered question |
| `qtype` | `noul`, `choice` or `score` |
| `options` | answer options; `["No", "Yes"]` for noul, ordered for score, shuffled for choice |
| `answer` | index of the right option (`answer_text` gives the option) |
| `ordinal` | `True` for score questions (options are ordered: ordinal loss possible) |
| `item`, `template_id`, `phrasing_id`, `concept` | where the question comes from |
| `flags` | quality flags of the entries behind the answer ([list](sampling.md#quality-flags)) |
| `evidence` | the report sentence(s) behind the answer, for debugging |

`qa.to_dict()` gives a plain dict.

### Another sampling config for one call

```python
gen.sample_qa_from_item(123, 8, {"qtype": {"noul": 0.2, "choice": 0.4, "score": 0.4}})
gen.set_sampling_config({"rarity": {"alpha": 1.0}})   # change the default (e.g. per epoch)
```

A config given to one call is merged over the default one, compiled once (a fraction of a
second) and cached. See [Sampling configuration](sampling.md).

## From the command line

```bash
E=runs/chest_ct_findings/gpt-oss-120b/ctrate_train_v31/exports

# candidates per template, with the answer distribution
python -m rsx qa stats $E
python -m rsx qa stats $E --by-concept --template noul_presence    # per concept
python -m rsx qa stats $E --out qa_stats.csv

# sample and print questions (right answer in [brackets])
python -m rsx qa preview $E --n-items 5 --k 8 --evidence
python -m rsx qa preview $E --item train_10001 --phrasings heldout

# shortcut check: accuracy without looking at the image
python -m rsx qa blind $E --weighted
```

All `qa` commands take `--dataset` (generator name, default `ct_rate`) and `--qa-templates`
(template set, default `chest_ct_findings`).
