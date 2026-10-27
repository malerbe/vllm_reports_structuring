# vllm_reports_structuring

Structure free-text radiology reports into **one fixed, typed template** with an LLM served by
[vLLM](https://docs.vllm.ai) on an H200 (NHR@FAU Helma), using **JSON-schema constrained decoding**
and many requests in parallel. Built for CT-RATE (finding-centric chest CT template), but the
template, the dataset reader and the model are each one variable at launch.

Inspired by [RadFields](../labels_extraction/RadFields): same template idea (YAML -> Pydantic ->
JSON schema), but without the template-selection steps, with typed fields (enum, number, integer,
boolean), async parallel requests, resumable jobs, and English prompts.

```
reports.csv
  -> feeder (one class per dataset)        report units: text + member ids (e.g. both CT-RATE recons)
  -> dedup by text, shard, select          one request per distinct report
  -> template (YAML -> Pydantic -> schema) system prompt (with the annotated template) + JSON schema
  -> async client (N in flight)  <->  vLLM in Apptainer, 127.0.0.1, same Slurm job, one H200
  -> runs/<template>/<model>/<run>/results.jsonl   append-only, one line per report, resumable
  -> rsx export                            parquet/csv tables: reports, findings, negations, ...
```

This repo only **extracts facts**. Generating question/answer pairs from them is a separate,
later step that reads the exported tables.

Contents: [Setup](#setup) · [Quick start](#quick-start) · [Running at scale](#running-at-scale) ·
[Interactive use](#interactive-use) · [Templates](#templates) · [Feeders (datasets)](#feeders-datasets) ·
[Models](#models) · [Outputs](#outputs) · [Architecture](#architecture) · [Tuning](#tuning) ·
[Troubleshooting](#troubleshooting) · [Licences](#licences)

## Setup

Everything runs from the **login node** (internet through `~/proxy.sh`); jobs on GPU nodes are
fully offline. All paths are in [`configs/paths.env`](configs/paths.env).

| What | Where | Why |
|---|---|---|
| code, run outputs (`runs/`) | `$HOME/projects/vllm_reports_structuring` | mounted on GPU nodes; outputs are small |
| Python client `rsx` | conda env `python311` (`PYTHON=` in paths.env) | already configured on this account |
| vLLM | one Apptainer image `.sif` on `/hnvme/.../containers/` | single file: no inode cost, pinned version |
| model weights | `/hnvme/workspace/y106bb17-lmalerba_data/models/` | ~30 files per model |
| download / build caches | login node `/scratch/$USER/rsx` | never on `/hnvme` (inode quota) or `$HOME` |
| vLLM / torch.compile caches | job-local `$TMPDIR` (deleted at job end) | set `RSX_VLLM_CACHE` (in paths.env) to keep them |

```bash
cd ~/projects/vllm_reports_structuring
scripts/install.sh --test          # pip install -e . into python311 (+ tests, ruff)
scripts/pull_vllm_image.sh         # vllm/vllm-openai:v0.31.0 -> .sif (~8 GB, a few minutes)
scripts/download_models.sh         # gpt-oss-120b (61 GB) + gpt-oss-20b (13 GB) + tiktoken files
python -m rsx models               # check: both models "ok", image "ok"
```

All three scripts are idempotent (rerun = check / resume).

- `install.sh` only adds what `python311` lacks (openai, pydantic, rapidfuzz, pytest, ruff). It pins
  torch, transformers, numpy, pandas, pyarrow and huggingface_hub to their current versions, and
  aborts if any of them would change.
- `download_models.sh` refuses to run when `/hnvme` is close to its inode quota (`RSX_HNVME_MAX_FILES`).
  It excludes the duplicate `original/` and `metal/` weights of gpt-oss and checks that every shard
  in the index is present.

Jobs are submitted with `--export=NONE`: settings must go in `configs/paths.env` (or
`configs/models.yaml`, the template), not in variables exported in your shell.

Do not edit the code (`src/rsx`) while jobs are running: the install is editable, so a job that
starts or imports a module during an edit can mix two versions. Templates are safe to edit, because
each run keeps its own snapshot and refuses to resume with a different template.

`python` below means the `python311` interpreter
(`/home/hpc/y106bb/y106bb17/conda_environments/python311/bin/python`, or `conda activate python311`).

## Quick start

```bash
# 1. Look at what would be sent (no GPU): annotated template, rules, the longest report
python -m rsx run --dry-run -t chest_ct_findings -m gpt-oss-20b -i ../validation_reports.csv | less

# 2. GPU smoke test: gpt-oss-20b, 20 random reports, partition preempt, 30 min
scripts/smoke_test.sh
tail -f runs/logs/rsx_smoke_<jobid>.log
scripts/check_run.sh runs/chest_ct_findings/gpt-oss-20b/smoke 18     # >= 18/20 ok + exports

# 3. Full validation set with gpt-oss-120b (1557 distinct reports for 3039 volumes)
TEMPLATE=chest_ct_findings MODEL=gpt-oss-120b INPUT=../validation_reports.csv scripts/launch.sh --then-export
```

**Measured on Helma** (H200, vLLM 0.31.0, gpt-oss-20b, 20 validation reports, reasoning effort low):

- server ready in about 160 s, then 20 reports in about 20 s (~3,500 reports/h with only 20 in
  flight), and the whole job in about 4 min;
- 20/20 valid, evidence quotes 94–99 % verbatim;
- reasoning effort `medium` made the 20b 8× slower and truncated 3/20 at 8192 tokens, with no
  clear gain, so the default stays `low`.

Not measured yet: the full validation set and gpt-oss-120b. Check the content of the first run by
hand: the 20b still makes fine-grained errors (e.g. pericardial vs pleural thickening, scope sites
of negations).

`scripts/launch.sh` is the one-line launcher. The variables are `TEMPLATE`, `MODEL`, `INPUT`,
`PARTITION` (h200 by default), `TIME`, `ARRAY`, `DEPENDENCY`, `JOB_NAME` and `DRY_RUN`. Extra
arguments go to `rsx job`. It calls `rsx submit`, which prints the exact `sbatch` command and the
log path.

To switch template or model, change the variable: `TEMPLATE=chest_ct_simple`, `MODEL=gpt-oss-20b`,
or `TEMPLATE=/path/to/my_template`.

## Running at scale

**Resume.** A run is identified by `runs/<template>/<model>/<run-name>/` (`--run-name` defaults to the
input file name). Each finished report is appended to `results.jsonl` immediately, so resubmitting the
*same command* skips what is done. This covers time limits, `scancel`, and preemption. Preemption on
`preempt` cancels the job (no requeue): resubmit it.

- Reports recorded as failed (invalid JSON, truncated...) are retried on the next run by default.
  Use `--no-retry-failed` to keep them as they are.
- A run refuses to resume if the template changed. The template hash covers prompts, schema and
  config, and is stored in every record. Either pick a new `--run-name`, or pass
  `--allow-template-change` to redo the reports made with the old version.
- When nothing is pending, the job exits without starting the GPU server.

**Stopping cleanly.** The job script has `#SBATCH --signal=B:TERM@300`. Five minutes before the time
limit, or on `scancel`/preemption, no new request starts. In-flight requests get `--drain-timeout`
seconds (120 by default) to finish and be written, then vLLM is stopped.

**Exit codes** (`sacct -j <id>`):

| Code | Meaning |
|---|---|
| 0 | everything selected is done |
| 3 | work remains (time limit, signal): resubmit |
| 4 | circuit breaker: the server stopped answering, or a fatal client error |
| 5 | the vLLM server failed to start (see `jobs/<id>/vllm.log`) |

**Longer than 24 h or faster.** You can chain jobs or split the work across GPUs:

```bash
# chain: the second job starts when the first ends (any state) and resumes
TEMPLATE=chest_ct_findings MODEL=gpt-oss-120b INPUT=train.csv TIME=24:00:00 scripts/launch.sh
DEPENDENCY=afterany:<jobid> TEMPLATE=chest_ct_findings MODEL=gpt-oss-120b INPUT=train.csv TIME=24:00:00 scripts/launch.sh
# parallel: 4 array tasks = 4 GPUs, each one deterministic shard (results.s<i>of4.jsonl)
ARRAY=4 TEMPLATE=chest_ct_findings MODEL=gpt-oss-120b INPUT=train.csv scripts/launch.sh
python -m rsx export runs/chest_ct_findings/gpt-oss-120b/train      # merges all shard files
```

Keep the same `ARRAY` value when you resubmit a sharded run (shard membership depends on it).

- **Logs:** one per task, `runs/logs/<job-name>_<arrayid>_<task>.log`.
- **Rerunning one shard:** `sbatch --array=2 ... scripts/run.sbatch job ... --num-shards 4` (the
  array index is then the shard id).
- **`--then-export` with shards:** only the last shard to finish exports (the others log that
  units are still missing). Each export file is written atomically.

**Subsets.** Use `--limit N` (first N), `--sample N --seed S` (random, deterministic), or
`--ids <unit ids | scan keys | VolumeNames>`. Reports are sent longest first, which shortens the tail
of the run.

## Interactive use

```bash
salloc -p h200 --gres=gpu:h200:1 -t 01:00:00
python -m rsx serve -m gpt-oss-20b      # waits for /health, writes runs/servers/<jobid>.json
# second shell on the same node:  srun --jobid=<jobid> --overlap --pty bash
python -m rsx run --server runs/servers/<jobid>.json -t chest_ct_findings -i ../validation_reports.csv \
    --limit 5 --run-name try --keep-reasoning
python -m rsx export runs/chest_ct_findings/gpt-oss-20b/try
```

The connection file holds the per-job API key (mode 600). `rsx run --base-url URL --api-key K`
connects to any OpenAI-compatible server.

## Templates

A template is a folder `templates/<name>/` (or any path):

| File | Required | Content |
|---|---|---|
| `template.yaml` | yes | the fields (format below) |
| `system_prompt.md` | no | instructions; placeholders `{{reader_view}}` (the annotated template) and `{{missing_value}}`. Default: `templates/_default/` |
| `user_prompt.md` | no | must contain `{{report}}`. Default: `templates/_default/` |
| `config.yaml` | no | default feeder, request overrides, evidence fields, export tables, postprocess hook |
| `postprocess.py` | no | `postprocess(table, df) -> df`, applied by `rsx export` |

A single `.yaml` file also works: it is treated as a RadFields template with the default prompts, e.g.
`-t ../labels_extraction/RadFields/templates/de/ct/ct_thorax_standard.yaml`.

Check a template with `python -m rsx template check|show|fields|schema <name>`.

### `template.yaml` format

The RadFields format is kept unchanged (all 147 RadFields templates load; see the tests):

```yaml
technique:                      # mapping  -> section (nested object)
  protocol: "<native or contrast>"   # string -> free-text field; the string is the hint shown to the LLM
sequences: ["T1", "T2"]         # list     -> repeated entries, element type = first item
```

On top of that, **typed leaves**: a mapping with the reserved key `_type`.

```yaml
contrast: {_type: enum, values: [non_contrast, contrast_enhanced], _hint: "From TECHNIQUE."}
slice_thickness_mm: {_type: number}
count_number: {_type: integer, _hint: "Only an explicit number."}
comparison_with_prior: {_type: boolean}
clinical_indication: {_type: string}
status: {_type: enum, values: [present, uncertain], _nullable: false}
size:                           # a section can be nullable as a whole
  _nullable: true
  _hint: "null unless a numeric measurement is given"
  numbers:
    - {_type: number}           # typed leaves inside lists are not nullable by default
_vocab:                         # root only, ignored: define YAML anchors (&name) reused with *name
  sides: &sides [left, right]
```

Rules:

- **`null` = "not mentioned in the report".** Typed leaves are nullable by default (`_nullable: false`
  forces a value). Lists use `[]` for "nothing". Text fields receive the template's `missing_value`.
  If a template needs an explicit "absent" answer, put it in an enum or model it as a separate list
  (as `negations` does in `chest_ct_findings`). This keeps *unknown* and *no* apart.
- Every key is required in the JSON schema (`additionalProperties: false`). Constrained decoding
  therefore always emits every key, in YAML order: put `evidence` first so it grounds the entry.
- Reserved keys (`_type`, `values`, `_hint`, `_nullable`, `_vocab`) are validated. Typos and
  unquoted YAML booleans (`yes`/`no`/`on`/`off` in `values`) fail with the path of the node.
- The generated schema only uses keywords that the xgrammar backend supports (`type`, `properties`,
  `required`, `additionalProperties`, `items`, `enum`, `anyOf`). `rsx template check` verifies this.

### `config.yaml`

```yaml
description: "..."
feeder: ct_rate                 # default dataset reader (--feeder overrides)
feeder_options: {}              # merged with --feeder-opt
missing_value: "not mentioned"  # for text leaves
reader_view_style: annotated    # or radfields (json.dumps of the YAML, as RadFields)
request: {temperature: 0.0}     # overrides of the model defaults (models.yaml); the CLI overrides both
evidence_fields: [evidence]     # verified against the report text at export
export_tables: {findings: findings, negations: negations}   # table name -> top-level list key
postprocess: postprocess.py
```

### The `chest_ct_findings` template

This is the v1 template, finding-centric, for CT-RATE English reports. The vocabularies were taken
from the validation reports.

- `exam`: contrast, slice thickness, clinical indication, comparison with prior, regions not
  optimally evaluable.
- `findings[]`: one entry per finding.
  - `evidence`, `finding_type` (an enum of 45: the 18 CT-RATE labels, frequent extras, `other`),
    `name`, `status` (present | uncertain).
  - Location: `region`, `laterality`, `lobes[]`, `sites[]` (lobe-qualified segments, lymph node
    stations, vessel parts), `distribution[]`.
  - Count: `count`, `count_number`.
  - `size`: numbers and unit **as written**, measure, qualifier (exact, up_to, range...), is_largest.
  - `severity` (minimal..severe, only from explicit words), `change` (vs prior), `descriptors[]`.
- `negations[]`: explicit absence or normal statements.
  - `finding_type`, `kind` (explicit_negation | normal_statement for a fixed list of mappings).
  - `qualifier` ("suspicious", "significant"... = restricted negation).
  - Scope: `regions[]`, `laterality`, `sites[]`.
- `impression_diagnoses[]`: diagnosis, category, certainty (definite | probable | possible).

The prompt explains the splitting rules, the size rules ("millimetric" is not a measurement), the
negation mappings, the Turkish-translation wording ("recessions", "calibration", "natural", "CTO")
and gives a glossary of finding types.

At export, `postprocess.py` adds `size_mm_max`/`size_mm_min`: mm and cm are converted, while `m`
and `none` give null plus a warning. `label_mapping.yaml` maps finding types to the 18 CT-RATE
labels.

For downstream question generation, the intended reading is:

- **yes**: a `findings` entry with status `present` (`uncertain` is not yes).
- **no**: a `negations` entry with `qualifier: null` whose scope covers the question.
- anything else is **unknown**.

### Writing a new template

1. `mkdir templates/my_template` and write `template.yaml` (start from `chest_ct_simple`, the minimal
   example).
2. Optionally add `system_prompt.md`, `user_prompt.md`, `config.yaml` (at least `feeder:`) and
   `postprocess.py`.
3. `python -m rsx template check my_template`, then
   `python -m rsx run --dry-run -t my_template -i data.csv`.
4. `TEMPLATE=my_template MODEL=gpt-oss-20b INPUT=data.csv PARTITION=preempt TIME=00:30:00 scripts/launch.sh --sample 20 --run-name try`.

## Feeders (datasets)

A feeder turns one input file into *report units* (`key`, `text`, `members`). `members` are the ids
the result applies to; exports write one row per member. `python -m rsx feeders` lists them with
their options.

- **`ct_rate`** (`CTRATEFeeder`) reads a CT-RATE reports CSV.
  - The text is `CLINICAL INFORMATION / TECHNIQUE / FINDINGS / IMPRESSION` blocks; empty columns are
    skipped.
  - Reconstructions `<scan>_<n>.nii.gz` of one scan are grouped (they share the report). A group
    whose texts differ is split, with a warning.
  - Options: `id_column`, `sections` (`[[column, heading], ...]`, e.g. to use other columns),
    `group_reconstructions`, `group_regex`.
- **`csv`** (`GenericCSVFeeder`) handles any CSV:
  `--feeder csv --feeder-opt id_column=report_id --feeder-opt "text_columns=[findings, impression]"`.

After the feeder, units with **identical text** are merged (one request, all members), unless you pass
`--no-text-dedup`. CT-RATE validation: 3039 volumes -> 1564 scans -> 1557 distinct texts.

**Adding a dataset** means one file and one import:

```python
# src/rsx/feeders/my_dataset.py   (+ import it in src/rsx/feeders/__init__.py)
from typing import Any, ClassVar
import pandas as pd
from .base import ReportFeeder, ReportUnit, join_sections, register_feeder


@register_feeder("my_dataset")
class MyDatasetFeeder(ReportFeeder):
    """One line description (shown by `rsx feeders`)."""

    member_column = "StudyID"
    default_options: ClassVar[dict[str, Any]] = {"sep": ";"}

    def iter_units(self):
        df = pd.read_csv(self.input_path, sep=self.options["sep"], dtype=str, keep_default_na=False)
        for row in df.to_dict("records"):
            text = join_sections(row, [("Befund", "FINDINGS"), ("Beurteilung", "IMPRESSION")])
            yield ReportUnit(key=row["StudyID"], text=text, members=(row["StudyID"],))
```

## Models

[`configs/models.yaml`](configs/models.yaml) holds one entry per model:

- the HF repo, the local folder and the served name;
- the client `concurrency`;
- `vllm_args` (any `vllm serve` flag);
- `vllm_env` (extra variables inside the container);
- `request` defaults: `reasoning_effort`, `temperature`, `max_tokens`, `timeout_s`, retries.

To add a model, add an entry and run `scripts/download_models.sh <name>`. For non-gpt-oss models,
remove `reasoning-parser: openai_gptoss` and `reasoning_effort`, and use `request.extra_body` for
model-specific options (e.g. `chat_template_kwargs`).

gpt-oss specifics (vLLM 0.31):

- The JSON schema constrains only the *final* channel, so the model still reasons freely first.
  `reasoning_chars` in each record shows that reasoning happened.
- `reasoning_effort` is sent explicitly (`low`, as in the RadFields paper).
- Harmony puts the current date in the system message. The server fixes it
  (`VLLM_SYSTEM_START_DATE`, `RSX_SYSTEM_DATE`), so requests stay deterministic and share the
  prompt prefix cache.

## Outputs

```
runs/<template>/<model>/<run>/
  run.yaml                 template, hash, model, input, feeder + options, request params, list of jobs
  template/                snapshot: template.yaml, prompts, config, schema.json, rendered_system_prompt.md
  results.jsonl            one JSON line per report (results.s<i>of<n>.jsonl for array jobs)
  jobs/<jobid>/            vllm.log, client.log, environment.json (GPU, driver, vLLM version), stats.json
  exports/                 rsx export (or --then-export)
```

A record (`rsx.results.Record`) holds:

- identity: `id` (sha1 of the text), `keys`, `members`, `text`;
- outcome: `status`, `attempts`, `parsed` (the JSON), `raw` (only on failure), `finish_reason`,
  `error`;
- generation: `reasoning_chars` (and `reasoning` with `--keep-reasoning`), `usage` (tokens),
  `latency_s`;
- provenance: `model`, `template`, `template_hash`, `vllm_version`, `job_id`, `ts`.

`status` is one of:

| Status | Meaning |
|---|---|
| `ok` | valid JSON, validated by the Pydantic model |
| `invalid_json`, `validation_error`, `empty` | the model's output was unusable |
| `truncated` | `max_tokens` was reached |
| `request_error` | HTTP 400, e.g. prompt too long |

Model failures are retried once with another seed at `temperature=0.7`, then recorded. Server errors
(connection, 5xx, timeout) are retried with backoff and are **not** recorded, so the next run redoes
them.

`python -m rsx export RUN_DIR` writes `exports/` (parquet + csv):

| File | Content |
|---|---|
| `reports` | one row per member (e.g. VolumeName): status, scalar fields as `section.field`, `n_<list>` counts, lists as JSON |
| `findings`, `negations`, `diagnoses` | from `export_tables`: one row per item × member, nested fields flattened, `evidence_verbatim` / `evidence_normalized` / `evidence_fuzzy` checks |
| `failures.csv` | status, error and raw output of failed reports |
| `summary.json` | statuses, latencies, tokens, evidence match rates |

Typed `null` stays `NA`; it is never turned into `False` or `""`. `--no-explode` gives one row per
report instead of one per member. `python -m rsx stats RUN_DIR` prints the summary only.

## Architecture

| Module | Role |
|---|---|
| `rsx.paths` | `configs/paths.env` parser (same file is sourced by bash) |
| `rsx.registry` | model registry, request defaults (model -> template -> CLI) |
| `rsx.template` | YAML -> IR (`TextLeaf`, `TypedLeaf`, `Section`, `ListOf`) -> Pydantic model -> vLLM schema; reader view; hash; snapshot |
| `rsx.feeders` | `ReportFeeder` base class + registry; `ct_rate`, `csv` |
| `rsx.data` | `WorkUnit` (text dedup, stable ids), sharding, selection |
| `rsx.client` | `AsyncOpenAI` + httpx (`trust_env=False`), worker pool, retries, error classes, drain on stop |
| `rsx.results` | `Record`, append-only `ResultsWriter`, resume (`pending_units`), `RunLock` |
| `rsx.server` | `VllmServer`: Apptainer command, env (`APPTAINERENV_*`, API key not in argv), port, `/health`, process-group stop |
| `rsx.job` | `rsx job` / `run` / `serve` / dry run: orchestration, signals, stats, environment |
| `rsx.export`, `rsx.evidence` | tables, evidence verification, postprocess hook |
| `rsx.stats` | progress (reports/h, tok/s, ETA) and run summaries |
| `rsx.download` | weights + tiktoken download (used by `download_models.sh`) |

Inside a job (`scripts/run.sbatch` -> `python -m rsx job ...`):

1. The input is loaded, deduplicated, sharded and selected. Pending units are computed from
   `results*.jsonl`, under a file lock on the run.
2. `apptainer exec --nv --cleanenv` runs `vllm serve` on `127.0.0.1:<port>`. The port is derived
   from the job id, because h200 nodes are shared by 4 jobs.
   - A random API key is passed through the environment.
   - The server is offline (`HF_HUB_OFFLINE`, no usage stats), with caches in `$TMPDIR`.
   - Flags: `--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'`,
     `--reasoning-parser openai_gptoss`, `--async-scheduling`.
3. The job polls `/health`, failing early if the process dies (the log tail is printed).
4. The async client keeps `concurrency` requests in flight. Each request carries
   `response_format={"type": "json_schema", ...}`, `reasoning_effort`, `max_completion_tokens`,
   `temperature` and `seed`. Each answer is validated by the Pydantic model and written as one line.
5. At the end, or on a signal, in-flight requests are drained, vLLM is stopped (TERM, then KILL of
   the process group), `stats.json` is written, and the export runs if `--then-export` was given.

## Tuning

- **`concurrency`** (models.yaml or `--concurrency`): the number of requests in flight.
  - vLLM batches them continuously; keep it at or below `--max-num-seqs`.
  - 256 is the default; watch `progress ... gen tok/s` in the log.
- **`max-model-len`**: 16384 by default. The system prompt of `chest_ct_findings` is about 4.2k
  tokens and the longest report about 0.6k; leave room for `max_tokens` (8192).
- **`max_tokens`**: if `truncated` records appear, raise it (and `max-model-len`).
- **`temperature`**: 0.0 by default, for reproducible extraction. OpenAI recommends 1.0 for gpt-oss;
  compare the two on ~50 reports if quality matters more than determinism.
- **Prefix caching** is on: the system prompt is identical for every report, so it is computed once.
- **Faster startup**: `RSX_VLLM_CACHE=${HOME}/.cache/rsx-vllm` in `configs/paths.env` keeps the
  torch.compile cache between jobs.

## Troubleshooting

- **Exit 5 / "vLLM exited"**: read `runs/.../jobs/<id>/vllm.log`.
  - Driver too old for CUDA 13: use the image tag `v0.31.0-cu129`.
  - Out of memory: lower `gpu-memory-utilization` or `max-num-seqs`.
- **Many `truncated`**: raise `max_tokens`; check `reasoning_chars` (long reasoning loops).
- **Many `validation_error`**: should not happen with constrained decoding.
  - Check `vllm.log` for grammar warnings (backend fallback).
  - Run `rsx template schema T`.
- **Template paths**: `rsx submit` / `launch.sh` validate the template on the login node and pass it
  to the job as an absolute path, so relative paths (`./my_template`) work from any directory.
- **`TemplateChangedError`**: the template, prompts or config changed since the run started. Use a
  new `--run-name` or `--allow-template-change`.
- **"is locked by another process"**: another job is writing this run (or shard). Wait, or use
  another run name.
- **Inode quota**: `find /hnvme/workspace/y106bb17-lmalerba_data | wc -l` (about 61k soft limit).
  Only the SIF and weights live there.
- **Tests**: `python -m pytest` (CPU only, a fake OpenAI server) and `python -m ruff check src tests`.

## Licences

| Component | Licence |
|---|---|
| this code | MIT |
| vLLM | Apache-2.0 |
| gpt-oss weights | Apache-2.0 (+ OpenAI usage policy) |
| RadFields (template format) | MIT |
| CT-RATE reports | CC BY-NC-SA 4.0 |

Derived data (extracted facts, question/answer sets) inherit the non-commercial, share-alike terms.
