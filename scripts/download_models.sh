#!/bin/bash
# Download model weights (configs/models.yaml) + tiktoken files to RSX_MODELS_DIR. Login node.
#   scripts/download_models.sh                       all models in the registry
#   scripts/download_models.sh gpt-oss-20b           only some
#   scripts/download_models.sh --check gpt-oss-20b   check completeness only
# Idempotent: complete models are skipped, interrupted downloads resume.
# The HF cache goes to the login node's /scratch; only the final files land on /hnvme
# (about 30 files per model). The script refuses to start if /hnvme is close to its inode quota.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_login_node

if [[ "${1:-}" != "--check" ]]; then
    n=$(count_files "$RSX_DATA")
    log "files in $RSX_DATA: $n (guard: $RSX_HNVME_MAX_FILES)"
    (( n + 200 < RSX_HNVME_MAX_FILES )) || die "too close to the /hnvme inode quota"
    use_proxy
fi

export HF_HOME="$RSX_SCRATCH/hf"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$RSX_MODELS_DIR"
# PYTHONPATH: works before `pip install -e .` (only needs pyyaml, pydantic, huggingface_hub).
PYTHONPATH="$RSX_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m rsx.download "$@"

[[ "${1:-}" == "--check" ]] || log "files in $RSX_DATA now: $(count_files "$RSX_DATA")"
