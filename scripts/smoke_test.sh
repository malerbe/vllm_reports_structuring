#!/bin/bash
# GPU smoke test: gpt-oss-20b on 20 random CT-RATE validation reports, partition preempt, 30 min.
#   scripts/smoke_test.sh [INPUT] [extra rsx job args]
# Then: scripts/check_run.sh runs/chest_ct_findings/gpt-oss-20b/smoke 18
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INPUT=${1:-$(dirname "$RSX_ROOT")/validation_reports.csv}
shift || true
exec "$PYTHON" -m rsx submit --template chest_ct_findings --model gpt-oss-20b --input "$INPUT" \
    --partition preempt --time 00:30:00 --job-name rsx_smoke \
    -- --sample 20 --seed 0 --run-name smoke --keep-reasoning --then-export "$@"
