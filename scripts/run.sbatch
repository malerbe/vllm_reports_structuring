#!/bin/bash -l
# One vLLM server (Apptainer) + async client on one H200. Arguments go to `rsx`:
#   sbatch scripts/run.sbatch job --template chest_ct_findings --model gpt-oss-120b --input /path/reports.csv
#   sbatch -p preempt --time 00:30:00 scripts/run.sbatch job ... --sample 20 --run-name smoke
# Easier: scripts/launch.sh or `rsx submit` (login node). Resubmitting the same command resumes the run.
# Exit codes: 0 done, 3 work remaining (timeout / preemption: resubmit), 4 circuit breaker, 5 server failed.
#SBATCH --job-name=rsx
#SBATCH --partition=h200
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/hpc/y106bb/y106bb17/projects/vllm_reports_structuring/runs/logs/%x_%j.log
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

RSX_ROOT=/home/hpc/y106bb/y106bb17/projects/vllm_reports_structuring
set -a
source "$RSX_ROOT/configs/paths.env"
set +a
# Requests go to 127.0.0.1: never through a proxy.
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export HF_HUB_OFFLINE=1

echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION, TMPDIR=${TMPDIR:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd "$RSX_ROOT"
# exec: python receives SIGTERM directly (Slurm --signal, scancel, preemption) and stops cleanly.
exec "$PYTHON" -m rsx "$@"
