# Filippo: reaction and free-energy handoff

The public entry point is `reproducibility/reaction/README.md`. Use the original
production model in `models/fine_tuned_m3gnet_model`, not the pathway-adapted fit.

1. Start from `reproducibility/reaction/snapshots/ps_330ps.data`: complete periodic
   3445-atom coordinates, cell, masses, image flags and saved velocities.
   Binary restart and ASE extxyz/explicit velocity CSV are alongside it.
2. The 100 ps intact reference and 260/340 ps prereaction states are also supplied.
   Central S-S IDs: **3432, 3433**. Original LS01 Li IDs: **3436, 3437**.
   IDs are LAMMPS/PLUMED 1-based; ASE indices are one less after sorting by ID.
3. `continue_330ps.in` is a heating-continuation example. `monitor.plumed`
   proposes S-S and Li-S coordination CVs without bias. The archived binary
   lacks PLUMED/COLVARS: a suitably enabled build is required for enhanced sampling.
4. Verify supplied files with `python reproducibility/reaction/verify_snapshots.py`.
   See the AdvanceSoft provenance/build instructions under `reproducibility/lammps`.

Important interpretation: the event was near target 338 K during annealing,
with hydrogen-type mass 3.016049 amu. Choose and report the FES temperature and
mass convention explicitly; a new stationary 300 K FES is not a continuation of
the heating protocol. Lithium identities exchange, so an identity-invariant Li-S
coordination CV is preferable to assuming only the original two Li can associate.

The available static diagram comprises selected/relaxed vacuum clusters, not
NEB or a free-energy path. Please assess within-state fluctuations, competing
coordination environments, CV adequacy, convergence and uncertainty before
assigning an activation or reaction free energy. Current original-model cluster
force errors and unresolved energy normalization are in `revision_2026/results`.
The adapted potential was fit to the path points and cannot independently
validate the original model. No claim of thermodynamic favorability is retained
in the working revision solely from the two selected snapshots.

Needed return products: temperature/ensemble and CV/bias specification; sampling
lengths and convergence diagnostics; FES and uncertainty; state energy
distributions; exact model/checkpoint hashes; and a concise explanation of which
kinetic/thermodynamic claims the results do and do not support.
