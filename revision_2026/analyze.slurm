#!/bin/bash
#SBATCH --job-name=jpc_training_audit
#SBATCH --partition=roma
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=analysis_%j.out
#SBATCH --error=analysis_%j.err
set -euo pipefail
CODE=${REVISION_CODE:?}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mlip-revision-mpl-$USER
PYTHON=${MLIP_PROJECT_ROOT:-/sdf/group/mli/MLIP_Li_S_Fl_Elec}/miniconda3/bin/python
"$PYTHON" "$CODE/analyze_path.py"
"$PYTHON" "$CODE/analyze_training.py"
