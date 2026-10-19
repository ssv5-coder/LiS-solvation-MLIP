#!/bin/bash
#SBATCH --job-name=jpc_ml_density
#SBATCH --partition=ada
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ml_density_%j.out
#SBATCH --error=ml_density_%j.err
set -euo pipefail
source "${REVISION_CODE:?}/environment.sh"
python "$REVISION_CODE/density_ml.py" "${CHEMISTRY:?}"
