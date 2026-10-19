#!/bin/bash
#SBATCH --job-name=jpc_dft_pilot
#SBATCH --partition=roma
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=120
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=vasp_pilot_%j.out
#SBATCH --error=vasp_pilot_%j.err
set -eo pipefail
source /sdf/group/suncat/sw/rh8mvapich2-2.3.7-1/env.bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONHOME PYTHONPATH
PYTHON=${MLIP_PROJECT_ROOT:-/sdf/group/mli/MLIP_Li_S_Fl_Elec}/miniconda3/bin/python
"$PYTHON" "${REVISION_CODE:?}/run_vasp_pilot.py" "${PILOT_FOLDER:?}"
