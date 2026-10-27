# Sourced by the other scripts: loads configs/paths.env and defines helpers.
# shellcheck shell=bash

RSX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RSX_ROOT="${RSX_ROOT:-$(dirname "$RSX_SCRIPT_DIR")}"
set -a
# shellcheck source=/dev/null
source "${RSX_PATHS:-$RSX_ROOT/configs/paths.env}"
set +a

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "[$(date +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

# Internet on the login node goes through the NHR@FAU proxy.
use_proxy() {
    if [[ -f "$RSX_PROXY_SCRIPT" ]]; then
        # shellcheck source=/dev/null
        source "$RSX_PROXY_SCRIPT"
        log "proxy: ${https_proxy:-none}"
    else
        log "proxy script $RSX_PROXY_SCRIPT not found, using direct connection"
    fi
}

require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        die "run this on the login node (needs internet), not inside a Slurm job"
    fi
}

count_files() { find "$1" 2>/dev/null | wc -l; }
