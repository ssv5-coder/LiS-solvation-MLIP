#!/bin/bash
#SBATCH --job-name=jpc_replica
#SBATCH --partition=ada
#SBATCH --exclude=sdfada013
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --time-min=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=replica_%j.out
#SBATCH --error=replica_%j.err
set -uo pipefail
CODE=${REVISION_CODE:?}
source "$CODE/environment.sh"
python "$CODE/run_replica.py" --replica "${REPLICA_ID:?}" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
    wait "$child"
    status=$?
fi
if [ "$status" -eq 75 ]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit $?
fi
exit "$status"
