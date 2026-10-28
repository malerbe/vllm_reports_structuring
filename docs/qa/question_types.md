# Question types and answers

[← Q/A generation](README.md)

Every answer comes from the structured report: a **finding** (something described as present), a
**negation** (something said to be absent or normal), or the attributes of a finding. A question
whose answer the report does not give is never asked.

## Concepts

Questions are about **concepts**:

- a `finding_type` of the template (`lung_nodule`, `pleural_effusion`, ...);
- or `other:<name>` for a frequent `other` finding (`other:accessory spleen`): the name is
  normalized (lower case, no article, last word singular) and kept if it occurs in at least
  `other_min_count` reports (30, in [`concepts.yaml`](../../qa_templates/chest_ct_findings/concepts.yaml)).
  `exclude` removes them from the questions by default (see below).

`{finding}` in a question is one of the concept's display names (`concepts.yaml: display`), drawn
at random: *pleural effusion*, *fluid in the pleural space*.

## Yes, No, unknown

For each (report, concept):

| State | When | Answer |
|---|---|---|
| present | at least one finding with status `present` (not `change: resolved`) | Yes |
| negated | a usable negation, and no finding of the concept | No |
| not mentioned, closed world | the concept is in `closed_world` (or its `closed_world_if` condition holds) | No, flag `closed_world` |
| unknown | not mentioned otherwise, uncertain, or only negated with a narrowing qualifier | **no question** |

- **Uncertain findings** (`status: uncertain`, 'cannot be excluded') make the concept unknown:
  no Yes, and no No either (not even from an explicit negation or the closed world).
- **Present wins.** A finding and a negation of the same concept in one report (often 'previous
  nodules are no longer seen' + another nodule) give Yes. `rsx qa stats` counts these conflicts.
- **Parent concepts.** `implies` in `concepts.yaml` lists kinds of a broader concept:
  ground-glass or consolidation present blocks the No of `lung_opacity` ('no infiltration'), on
  the report and on the side where it is; an uncertain child makes the parent unknown.
- **Qualified negations.** 'No **suspicious** nodule' does not mean 'no nodule': a negation with a
  qualifier is ignored, unless the qualifier is listed as neutral for its type
  (`neutral_qualifiers`, e.g. 'pathological size' for lymphadenopathy, 'significant' for all).
  Such a report stays unknown for the concept, even when the concept is in the closed world.
- **Lateralized negations** ('no effusion on the right') only answer the side questions, unless
  the closed world fills the rest.

## Closed world

A report describes what is there; it rarely lists what is not. Many findings are therefore never
negated (ground-glass: 7,729 Yes / 19 No on CT-RATE train), and Yes-only concepts teach the model
'concept → Yes'. The closed world makes "not mentioned" a No, for chosen types only:

- `closed_world: [...]`: everywhere. Chosen for findings that (1) a radiologist reliably reports
  when seen, (2) are well defined and (3) are always in the chest CT field of view.
- `closed_world_if: {type: {regions: [...], evidence: regex}}`: only in reports that have an entry
  (finding or negation, any type) in one of the regions, matching `evidence` if given, and no
  `other` finding there (an unnamed abnormality). For the partly imaged upper abdomen: the organ
  must have been looked at.

Both are empty in `concepts.yaml` and set per dataset. The CT-RATE
[overlay](../../qa_templates/chest_ct_findings/datasets/ct_rate.yaml) lists 30 chest types
(lung, pleura, heart and vessels, mediastinum, bone lesions and fractures, devices) and makes the
upper abdominal organs and degenerative spine changes conditional ('bone structures are natural':
4.9% of these reports describe degeneration, against 27.6% overall). Silence stays unknown for
thymic tissue, thyroid, breast / chest wall, post-surgical changes and lung cysts / bullae, which
are often left out of reports.

Every No made this way carries the `closed_world` flag
([flag weights](sampling.md#quality-flags)). Effect on CT-RATE train, blind accuracy (chance 0.5):
`noul_presence` 0.71 → 0.52, `noul_presence_side` 0.66 → 0.52; `choice_which_present` (chance
0.25) 0.60 → 0.31.

## Extraction fixes, excluded concepts

Reading the CT-RATE train reports against the extraction showed systematic errors of the
structuring template. `concepts.yaml` fixes them before the concepts are built:

| Rule | Example | Effect |
|---|---|---|
| `negation_rules` drop | 'mediastinal vascular structures normal' typed `arterial_wall_calcification` | 163 false No removed |
| `negation_rules` drop | esophageal 'no wall thickening' typed `peribronchial_thickening`; 'gallbladder not observed (operated)' typed `post_surgical_change`; abdominal 'no cystic mass' typed `lung_cyst_or_bulla`; 'the left kidney was not observed' | false No removed |
| `negation_rules` retype | 'no pathologically enlarged lymph node' typed `lymph_node_not_enlarged` | No of `lymphadenopathy` |
| `finding_rules` negate | 'millimetric lymph nodes' (`lymph_node_not_enlarged` finding, short axis ≤ 10 mm) | No of `lymphadenopathy` |
| `finding_rules` add | crazy paving (GGO) → `interlobular_septal_thickening`; fibroatelectasis → `atelectasis`; honeycomb / reticular (sequela) → `reticulation`; bronchial ectasia / wall thickening (`airway_abnormality`) → `bronchiectasis` / `peribronchial_thickening` | missed Yes added, flag `derived` |

Rules match the evidence with a regex (`evidence`, `evidence_not`); finding rules only start from
findings that are present and not resolved. `exclude` removes concepts
from every question: `other:*` (heterogeneous), `infection` (a diagnosis, not a visible sign) and
`lymph_node_not_enlarged` (a normality statement, turned into a No of lymphadenopathy).

## Attributes of a present concept

When a report has several entries of one concept (two nodules, effusion right + left), the
attributes are aggregated, and an attribute that the entries do not determine gives no question:

| Attribute | Rule |
|---|---|
| laterality | union (right + left = bilateral); undetermined if one entry has no laterality |
| lobe | the single lobe of all entries; undetermined if several lobes or an entry without lobe |
| region | the single region of all entries |
| severity | the value if the entries that give one agree, else undetermined |
| size (`size_mm_max`) | largest measurement; 'less than' / 'greater than' sizes are ignored |
| count | only if the concept has a single top-level entry (`subset_of` empty); `count_number >= 2` = multiple |

## Templates of `chest_ct_findings`

Counts on CT-RATE train (2026-10-28, `python -m rsx qa stats`):

| Template | Type | Question | Candidates | Answers |
|---|---|---|---|---|
| `noul_presence` | noul | Does this CT show any {finding}? | 748k | No 642k (461k closed world) / Yes 106k |
| `noul_presence_side` | noul | Is there any {finding} on the {side} side? | 573k | No 485k (384k closed world) / Yes 88k |
| `choice_laterality` | choice | On which side is the {finding}? | 66k | right / left / both sides |
| `choice_lobe` | choice | In which lobe is the {finding} located? | 11k | 5 lobes (lingula = left upper) |
| `choice_which_present` | choice | Which of the following findings is present? | 106k | 1 present + 3 negated in the same report |
| `score_severity` | score | How severe is the {finding}? | 10k | minimal < mild < moderate or severe |
| `score_count` | score | Is there one {finding} or more than one? | 17k | one < more than one |
| `score_nodule_size` | score | What is the size of the largest {finding}? | 5.7k | 6 bins, < 4 mm ... ≥ 30 mm |
| `score_effusion_thickness` | score | What is the maximal thickness of the {finding}? | 1.7k | 5 bins |
| `score_aortic_diameter` | score | What is the maximal diameter of the dilated aorta? | 1.9k | 4 bins |

`choice_which_present` shows the present concept among concepts **negated in the same report**:
the distractors are known to be absent. A concept that is never negated could never be a
distractor, and seeing it among the options would give the answer away; see
[`which_present`](sampling.md#choice_which_present) and the [shortcut check](sampling.md#shortcut-check).

The CT-RATE overlay merges `moderate` and `severe` (142 and 177 cases in train) into one level;
other datasets keep the 4 levels of `templates.yaml`.

Questions about the change since a prior exam are not generated: the model does not see the
prior exam.
