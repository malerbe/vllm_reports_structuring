#!/bin/bash
# Build the vLLM Apptainer image (one .sif file on /hnvme) from the pinned Docker image. Login node.
#   scripts/pull_vllm_image.sh            skip if the .sif already exists
#   scripts/pull_vllm_image.sh --force    rebuild
# Apptainer's cache and temp dirs go to the login node's local /scratch: unpacking the image
# creates ~100k small files, which must never land on /hnvme (inode quota) or $HOME.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_login_node

if [[ -f "$RSX_SIF" && "${1:-}" != "--force" ]]; then
    log "$RSX_SIF already exists (use --force to rebuild)"
else
    use_proxy
    export APPTAINER_CACHEDIR="$RSX_SCRATCH/apptainer-cache"
    export APPTAINER_TMPDIR="$RSX_SCRATCH/apptainer-tmp"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$RSX_SIF")"
    log "pulling $RSX_VLLM_IMAGE -> $RSX_SIF (about 10 GB, takes a while)"
    rm -f "$RSX_SIF.partial"
    apptainer pull "$RSX_SIF.partial" "$RSX_VLLM_IMAGE"
    mv "$RSX_SIF.partial" "$RSX_SIF"
    log "cleaning $APPTAINER_TMPDIR (the layer cache in $APPTAINER_CACHEDIR is kept for rebuilds)"
    rm -rf "$APPTAINER_TMPDIR"
fi

version=$(apptainer exec --cleanenv "$RSX_SIF" python3 -c 'import vllm; print(vllm.__version__)')
echo "$version" > "$RSX_SIF.version"
log "vLLM in image: $version ($(du -h "$RSX_SIF" | cut -f1))"
