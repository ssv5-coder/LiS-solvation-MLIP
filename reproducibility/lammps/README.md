# AdvanceSoft LAMMPS provenance

The reaction logs and binary checkpoint headers identify **LAMMPS 2 Aug 2023**.
The current source checkout at the absolute ML-GNNP path used by the inputs is:

- Upstream: https://github.com/advancesoftcorp/lammps
- Commit: `bd3766d4271c618950450bebdc9c7a5ff3333211` (2 April 2025)
- Local modifications: `local_changes.patch`
- Current driver snapshot: `gnnp_driver.py`
- Recovered build options: `build_options.txt`

These are the surviving source/build artifacts. The 2025 logs do not record
a Git commit, so this is not proof that every present source byte was identical
on the execution date. The LAMMPS release and checkpoint compatibility are
directly verified. A separate older checkout under the training folder is at
`c224fb644122e8c1c53b88a380fb9642c6ea1421`; the simulation inputs explicitly
point to the first checkout's ML-GNNP driver.

## Build

Install the repository's compatible Python environment first. On a machine
with CMake, a C++ compiler, CUDA and Python development libraries:

```bash
git clone https://github.com/advancesoftcorp/lammps.git advancesoft-lammps
cd advancesoft-lammps
git checkout bd3766d4271c618950450bebdc9c7a5ff3333211
git apply /path/to/LiS-solvation-MLIP/reproducibility/lammps/local_changes.patch
cmake -S cmake -B build \
  -D CMAKE_BUILD_TYPE=Release -D BUILD_MPI=OFF -D BUILD_OMP=ON \
  -D BUILD_SHARED_LIBS=ON -D PKG_ML-GNNP=ON -D PKG_PYTHON=ON \
  -D PKG_GPU=ON -D GPU_API=cuda -D GPU_ARCH=sm_89 -D GPU_PREC=mixed \
  -D PKG_EXTRA-COMPUTE=ON -D PKG_EXTRA-DUMP=ON -D PKG_EXTRA-FIX=ON \
  -D PKG_MANYBODY=ON -D PKG_MOLECULE=ON -D PKG_RIGID=ON \
  -D Python_EXECUTABLE=/absolute/path/to/environment/bin/python
cmake --build build -j 4
```

`sm_89` matches the recovered L40S build; choose the appropriate architecture
for another GPU. The historical build is serial (`BUILD_MPI=OFF`); run one
LAMMPS process. Ensure its shared library and the Python environment's libraries
are discoverable on the target machine. The original SLURM script is archived
in `../reaction/original_inputs/` as site-specific provenance.

The template uses `pair_style gnnp`, which lets the Python driver select a
GPU when available, and can also run on CPU. The original used `gnnp/gpu`.
For enhanced sampling, enable and link PLUMED or COLVARS in a separate build;
both were disabled in the original cache. No bias engine has been tested by
this handoff.

The repository requirements use the CPU DGL wheel for a portable CPU test.
GPU execution requires a DGL build matching the chosen CUDA/PyTorch stack.
The surviving local installation is DGL 2.1.0+cu118 with Torch 2.2.0;
its CUDA libraries must be available on compute nodes. The verified CPU
environment uses DGL 1.1.3. Do not expect a CPU-only DGL wheel to run on CUDA.

Do not add a second D3 correction at ML-MD runtime: the model was fitted to
PBE-D3 labels. The archived MatGL driver explicitly requests stress in eV/A^3.

The driver snapshot and patch retain AdvanceSoft/LAMMPS GPL licensing;
see `LICENSE.AdvanceSoft-LAMMPS`. The repository MIT license does not replace
that third-party license.
