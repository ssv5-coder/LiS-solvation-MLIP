#!/bin/bash
#SBATCH --job-name=jpc_revision_analysis
#SBATCH --partition=roma
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=postprocess_%j.out
#SBATCH --error=postprocess_%j.err
set -euo pipefail
unset PYTHONHOME PYTHONPATH
PYTHON=${MLIP_PROJECT_ROOT:-/sdf/group/mli/MLIP_Li_S_Fl_Elec}/miniconda3/bin/python
case "${ANALYSIS:?}" in
    density) "$PYTHON" "${REVISION_CODE:?}/analyze_density.py" ;;
    spin) "$PYTHON" "${REVISION_CODE:?}/analyze_spin.py" ;;
    replicas) "$PYTHON" "${REVISION_CODE:?}/analyze_replicas.py" ;;
    *) exit 2 ;;
esac
"$PYTHON" "$REVISION_CODE/collect_status.py"
