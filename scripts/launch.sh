#!/bin/bash
# One-line launcher (login node). Variables select the template, model and input; extra args go to `rsx job`.
#   TEMPLATE=chest_ct_findings MODEL=gpt-oss-120b INPUT=../validation_reports.csv scripts/launch.sh
#   TEMPLATE=chest_ct_simple MODEL=gpt-oss-20b INPUT=reports.csv PARTITION=preempt TIME=01:00:00 \
#       scripts/launch.sh --limit 100 --run-name test100
#   ARRAY=4 ... scripts/launch.sh      # 4 jobs in parallel (one shard each, 4 GPUs)
#   DRY_RUN=1 ... scripts/launch.sh    # no job: print the prompt of the longest report
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${TEMPLATE:?set TEMPLATE (a folder name in templates/ or a path)}"
: "${INPUT:?set INPUT (input file)}"
MODEL=${MODEL:-gpt-oss-120b}
PARTITION=${PARTITION:-h200}
TIME=${TIME:-04:00:00}
JOB_NAME=${JOB_NAME:-rsx_${TEMPLATE##*/}}

if [[ -n "${DRY_RUN:-}" ]]; then
    exec "$PYTHON" -m rsx run --dry-run --template "$TEMPLATE" --model "$MODEL" --input "$INPUT" "$@"
fi
submit=(--partition "$PARTITION" --time "$TIME" --job-name "$JOB_NAME")
[[ -n "${ARRAY:-}" ]] && submit+=(--array "$ARRAY")
[[ -n "${DEPENDENCY:-}" ]] && submit+=(--dependency "$DEPENDENCY")
exec "$PYTHON" -m rsx submit --template "$TEMPLATE" --model "$MODEL" --input "$INPUT" "${submit[@]}" -- "$@"
