# Reaction handoff for Filippo

Start with `snapshots/ps_330ps.data` (portable LAMMPS data with a Velocities
section) or `ps_330ps.restart` (original binary checkpoint).
Each snapshot contains the complete periodic 3445-atom electrolyte:
20 LiTFSI, 96 DME, 143 DOL, and two initially assigned Li2S6 units.
The production potential is the repository's
`models/fine_tuned_m3gnet_model`, not a pathway-adapted potential.

## Snapshot choices and timeline

| Time | Original step | Purpose |
|---|---:|---|
| 100 ps | 50000 | Earlier reference before the reported Li-departure state |
| 260 ps | 160000 | 91 ps before the reported central S-S extension near 351 ps |
| 330 ps | 230000 | 21 ps before extension; recommended starting checkpoint |
| 340 ps | 240000 | 11 ps before extension; closest supplied checkpoint |

The event belongs to pre-production heating, not the final 300 K production.
The initial 0-100000 steps used 0.002 ps; later heating used 0.001 ps.
Thus physical time after step 100000 is
`200 + (step - 100000)*0.001` ps.
The 200000-300000 restart segment ramps 325-350 K with Langevin damping
1 ps, seed 12345, and NPH at 1.01325 bar with pressure damping 5 ps.
At 330 and 340 ps the thermostat targets are 332.5 and 335 K.
Use the source inputs and manifests rather than inferring settings from folder
names or old comments. The later NVT production uses a 0.5 fs timestep.

Hydrogen-type atoms have the stored mass **3.016049 amu** in these checkpoints.
Changing this to protium or deuterium changes the dynamics.
Velocities are genuine saved restart velocities, not estimates from coordinate
differences or newly sampled Maxwell velocities.

## Atom indices

| Identity | LAMMPS / PLUMED ID (1-based) | ASE index (0-based, sorted by ID) |
|---|---|---|
| LS01 central breaking S-S pair | 3432, 3433 | 3431, 3432 |
| LS01 six sulfur atoms | 3430-3435 | 3429-3434 |
| LS01 originally retained Li | 3436 | 3435 |
| LS01 departing Li | 3437 | 3436 |
| LS02 six sulfur atoms | 3438-3443 | 3437-3442 |
| LS02 original Li atoms | 3444, 3445 | 3443, 3444 |

Original molecule membership is a label, not a constraint or charge assignment.
Other electrolyte lithium atoms can exchange with these sites.
The atom-type ordering is C F H Li N O S.
The data files preserve original IDs; use the `id` array in ASE rather than
assuming an arbitrary downstream format preserves ordering.
Distances and coordination CVs require minimum-image periodic distances.

## Files and usage

- `*.restart`: exact source checkpoints with atom state and available fix state.
- `*.data`: portable coordinates, masses, box, image flags, and velocities.
- `*.extxyz`: ASE-readable full-system coordinates, masses, IDs, and momenta.
- `*_velocities.csv`: explicit velocity components in Angstrom/ps by atom ID.
- `manifest.json`: times, geometry checks, kinetic temperatures, and source files.
- `checksums.json`: SHA-256 of each snapshot artifact.
- `continue_330ps.in`: reproduction-oriented heating continuation template.
- `monitor.plumed`: diagnostic central S-S and two Li-S coordination CVs, no bias.
- `original_inputs/`: historical inputs and successful restart lineage.

From this directory:

```bash
lmp -var model /absolute/path/to/models/fine_tuned_m3gnet_model \
    -var driver /absolute/path/to/advancesoft-lammps/src/ML-GNNP \
    -in continue_330ps.in
```

For a PLUMED-enabled build, add
`fix cv all plumed plumedfile monitor.plumed outfile plumed.log` before run.
The archived build had PLUMED and COLVARS disabled; enable one when building
the enhanced-sampling engine. These CVs are monitoring suggestions, not an FES
protocol. Temperature, bias parameters, equilibration, and convergence checks
for a new FES must be chosen separately.

For a portable restart replace `read_restart` with
`read_data snapshots/ps_330ps.data` and `reset_timestep 230000`.
This retains velocities but does not restore thermostat/barostat internal
variables. Even a binary restart does not preserve the Langevin random stream,
so neither route promises identical subsequent atom trajectories.

## Extraction

A CPU-only reader built from the matching 2 Aug 2023 LAMMPS sources read each
binary checkpoint. `pair_style zero 5.0` was used solely to satisfy
`write_data ... nocoeff` initialization. **No run, minimization, force-based
update, or velocity replacement occurred.** The exported zero pair is not the
potential for dynamics; the continuation explicitly reinstates M3GNet.
Original restart and data files are both provided for independent verification.
The extraction script is `package_checkpoints.py`.
The zero-step LAMMPS conversion template is `export_restart.in`.

Validate the supplied files with:

```bash
python verify_snapshots.py
```

Add `--ps-root /path/to/original/PS` to independently compare the coordinates
against the original dump files and successful restart branches.

Full production trajectories are not part of this handoff.
