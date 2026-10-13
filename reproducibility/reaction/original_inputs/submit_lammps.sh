#!/bin/sh
#SBATCH --job-name=smaller_ada
#SBATCH --output=jupyter_ada.out
#SBATCH --error=jupyter_ada.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # Launch 2 tasks, one per GPU
#SBATCH --cpus-per-task=4           # Each task gets 4 CPU cores (total 8 cores)
##SBATCH --gpus-per-node=1
#SBATCH --mem=200064M
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --partition=ada
#SBATCH --account=mli:default
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ssv5@slac.stanford.edu
#SBATCH --qos preemptable
##SBATCH --requeue
# Purge modules (and load any required ones if needed)
#module purge
# Set MPI fabric options (if required by your cluster configuration)
export I_MPI_FABRICS=shm:tmi
# Set the number of OpenMP threads per MPI rank
export OMP_NUM_THREADS=1
# Remove any stack limit (if needed by your application)
ulimit -s unlimited
# Source your bash initialization file (if it loads required paths, etc.)
#source ~/.bashrc
# Instead, source conda.sh to initialize conda for this session:
source /fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/etc/profile.d/conda.sh
conda activate lammps-m3gnet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/ml_Folder/advancesoft-lammps/build:$PATH"
# Ensure the conda environment's bin is in the PATH
export PATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/bin:$PATH"
# Optionally, if you run into issues with module imports, update PYTHONPATH
export PYTHONPATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib/python3.10/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
# Add CUDA 11 libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet/lib:$LD_LIBRARY_PATH
export PYTHONHOME=/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/miniconda3/envs/lammps-m3gnet
export LD_LIBRARY_PATH=$PYTHONHOME/lib:$LD_LIBRARY_PATH
# Verify the correct Python is being used
which python3
mpiexec -n $SLURM_NTASKS lmp -sf gpu -pk gpu 1 -in inp.lammps
