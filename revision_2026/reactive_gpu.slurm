#!/bin/bash
#SBATCH --job-name=jpc_reactive_check
#SBATCH --partition=ampere
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=reactive_%j.out
#SBATCH --error=reactive_%j.err
set -euo pipefail
source "${REVISION_CODE:?}/environment.sh"
python "$REVISION_CODE/reevaluate_path.py"
