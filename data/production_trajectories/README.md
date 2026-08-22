# Production trajectories

The final ASE trajectory files contain the 300 K coordinate sequences used for
the manuscript's large-system production analyses. The trajectory binaries are
not included in the current repository release and will be distributed
separately.

| File | Chemistry | Atoms | Frames | Cell length (Angstrom) |
|---|---|---:|---:|---:|
| `nonPS_production_300K.traj` | 20 LiTFSI / 96 DME / 143 DOL | 3429 | 23996 | 32.65455 |
| `PS_production_300K.traj` | non-PS system + 2 Li2S6 | 3445 | 24597 | 32.77722 |

The ML-MD integration timestep was 0.5 fs. Coordinates in these analysis-ready
bundles are effectively separated by 25 MD steps (12.5 fs). The files were
assembled from restarted production segments, and the original restart
boundaries are retained. Users should therefore inspect continuity at segment
boundaries before computing finite-difference velocities or other observables
that depend directly on adjacent frames. The manuscript analyses used the
documented reconstruction and analysis workflows for each observable.

Both systems were conditioned for 800 ps and then propagated at 300 K in the
NVT ensemble with a Nose-Hoover thermostat. See the manuscript and Supporting
Information for the complete annealing, pressure-control, and analysis
protocols.

Checksums will be published with the trajectory release.
