#!/bin/bash
# Source on a compute node. Do not modify the historical conda environment.
PROJECT=${MLIP_PROJECT_ROOT:-/sdf/group/mli/MLIP_Li_S_Fl_Elec}
ENV="$PROJECT/miniconda3/envs/lammps-m3gnet"
export PATH="$ENV/bin:$PROJECT/advancesoft-lammps/build:$PATH"
export PYTHONHOME="$ENV"
export PYTHONPATH="$ENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/advancesoft-lammps/build:$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for library in cuda_runtime cublas cusparse cuda_nvrtc cudnn cusolver curand cufft; do
    export LD_LIBRARY_PATH="$ENV/lib/python3.10/site-packages/nvidia/$library/lib:$LD_LIBRARY_PATH"
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mlip-revision-mpl-$USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -s unlimited
