#!/bin/bash
#SBATCH --job-name=jpc_pack_replicas
#SBATCH --partition=roma
#SBATCH --account=mli:default
#SBATCH --qos=preemptable
#SBATCH --constraint=OS_VER:9.6
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=pack_%j.out
#SBATCH --error=pack_%j.err
set -euo pipefail
export JULIA_NUM_THREADS=1
julia "${REVISION_CODE:?}/pack_replicas.jl"
