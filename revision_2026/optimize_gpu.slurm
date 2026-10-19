#!/bin/bash
#SBATCH --job-name=jpc_gpu_optimize
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=optimize_%j.out
#SBATCH --error=optimize_%j.err
set -euo pipefail
source "${REVISION_CODE:?}/environment.sh"
python "$REVISION_CODE/test_gpu_optimization.py"
