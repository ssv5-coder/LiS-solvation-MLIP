# Verification performed for the handoff (13 October 2026)

- Production model: all three serialized files match
  `big_boy_m3gnet_smaller_stress_ada_2_best` byte for byte.
- LAMMPS reader: built successfully from the recorded source checkout as
  a serial CPU-only 2 Aug 2023 executable. All four original checkpoints
  were read and exported without any integration or minimization.
- Source trajectory agreement: maximum coordinate differences for each
  checkpoint are approximately 0.00005 Angstrom, consistent with the
  original text dumps' rounding. Correct successful branches were used.
- IDs: 3445 unique IDs, ordered 1 through 3445 in every exported ASE file.
- Velocities: conversion from original LAMMPS Angstrom/ps to ASE and through
  extended XYZ differs by less than 1.71e-9 in ASE velocity units.
- Central S-S separations in the unrelaxed original states:
  2.06094, 2.01934, 2.12946 and 2.12748 Angstrom at 100, 260, 330 and 340 ps.
- 330 ps model/driver test: the archived Python driver initializes with
  a 5 Angstrom cutoff and stress enabled. On the full snapshot, CPU inference
  returns finite energy, forces, and stresses (energy 13.904296875 eV,
  maximum force norm 4.93873 eV/Angstrom). These are model outputs, not DFT
  validation. The original GPU log gives 13.835938 eV at this step;
  this CPU/source reconstruction is not bitwise identical.
- Training: one-batch training plus validation passed using official MatGL
  1.2.3, DGL 1.1.3 CPU, Torch 2.2.0, Lightning 2.5.0.post0 and
  TorchMetrics 1.3.2. The network has 288157 trainable parameters and zero
  frozen parameters. This verifies execution, not recovered scientific errors
  or completion of a 20-epoch retraining.
- The original installed DGL distribution is 2.1.0+cu118 and requires CUDA
  shared libraries. Temporary CPU test dependencies were used; the original
  simulation environment was not modified.
- The continuation input and diagnostic PLUMED definitions are templates;
  a complete continuation run or biased FES calculation was not performed.

The training-data audit and source limitations are documented separately in
`../training/README.md`. No full production trajectory is included.
