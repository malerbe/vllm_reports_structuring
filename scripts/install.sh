#!/bin/bash
# Install the rsx client into the existing conda env (PYTHON in configs/paths.env). Login node.
#   scripts/install.sh            install / update
#   scripts/install.sh --test     + run the unit tests and ruff
# Only missing packages are added (openai, pydantic, rapidfuzz, pytest, ruff...). The script
# checks that torch / transformers / numpy / pandas keep their versions. vLLM itself is NOT
# installed here: it runs from the Apptainer image (scripts/pull_vllm_image.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_login_node

[[ -x "$PYTHON" ]] || die "PYTHON=$PYTHON not found (edit configs/paths.env)"
log "python: $PYTHON ($("$PYTHON" --version 2>&1))"
use_proxy

guarded=(torch transformers numpy pandas pyarrow huggingface_hub)
versions() {
    "$PYTHON" - "${guarded[@]}" <<'EOF'
import importlib.metadata as md, sys
for name in sys.argv[1:]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}==absent")
EOF
}
before=$(versions)

export PIP_CACHE_DIR="$RSX_SCRATCH/pip-cache"
mkdir -p "$PIP_CACHE_DIR"
# Constraints pin the guarded packages to their current versions, so pip never up/downgrades them.
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
grep -v '==absent' <<<"$before" > "$constraints" || true
"$PYTHON" -m pip install --upgrade-strategy only-if-needed -c "$constraints" \
    -e "$RSX_ROOT[dev,download]"

after=$(versions)
# Only packages present before are guarded; absent ones may be installed (logged).
changed=$(comm -23 <(grep -v '==absent' <<<"$before" | sort) <(sort <<<"$after"))
if [[ -n "$changed" ]]; then
    diff <(echo "$before") <(echo "$after") || true
    die "guarded package versions changed (see diff above)"
fi
added=$(comm -13 <(sort <<<"$before") <(sort <<<"$after") | grep -v '==absent' || true)
[[ -n "$added" ]] && log "newly installed: $(tr '\n' ' ' <<<"$added")"
log "guarded packages unchanged: $(tr '\n' ' ' <<<"$after")"

mkdir -p "$RSX_RUNS_DIR/logs" "$RSX_RUNS_DIR/servers"
"$PYTHON" -m rsx --help >/dev/null && log "rsx CLI OK"

if [[ "${1:-}" == "--test" ]]; then
    cd "$RSX_ROOT"
    "$PYTHON" -m pytest
    "$PYTHON" -m ruff check src tests
fi
log "done"
