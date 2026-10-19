#!/bin/bash
#SBATCH --job-name=jpc_gpu_pilot
#SBATCH --partition=ada
#SBATCH --exclude=sdfada013
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
set -euo pipefail
CODE=${REVISION_CODE:?Set REVISION_CODE to the revision_2026 scripts directory}
source "$CODE/environment.sh"
nvidia-smi
python "$CODE/benchmark_gpu.py"
