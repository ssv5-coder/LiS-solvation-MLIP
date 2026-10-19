# JPC reviewer revision, 19 October 2026

This is a working, evidence-based revision, not a completed rebuttal. Original
models, installed packages, historical calculations and submitted manuscripts
are preserved. `common.py` defines the site defaults; override
`MLIP_PROJECT_ROOT` and `REVISION_WORK` when appropriate. Slurm files are SLAC
examples, not portable scheduler settings.

## Start here

- [Filippo handoff](FILIPPO_HANDOFF.md): selected complete periodic states,
  velocities, masses, indices, original engine and potential.
- [Training reconstruction](../training/README.md): full-network final-stage
  fine-tuning, initialization, logs, missing caches and split limitations.
- `results/`: compact audit outputs and provenance. No licensed PAW files,
  wavefunctions or full trajectories are released here.
- Working manuscript/SI and response document are in
  `JPCRevision/reviewer_revision_20261019/` outside this repository. They are
  explicitly provisional; final reviewer page/line references await pagination.

## Completed audits and unresolved scientific issues

1. Exported every S-S pair from the actual 120,000-frame `train.traj`, with
   minimum-image distances, frame/source provenance and separate initial
   polysulfide/TFSI sulfur membership. A short distance is not a bond assignment.
2. Checked 21 unrelaxed 148-atom cluster geometries/order and cached original-model
   predictions against raw converged DFT. Current force MAE/RMSE are
   185.84/382.95 meV/Angstrom. Raw energy MAE is 4646.64 meV/atom; relative-energy
   MAE is 1.3247 eV/cluster using the first frame as the shared reference.
   These are vacuum-cluster tests, not periodic-bulk reaction validation.
   `results/reactive_validation/summary.json` records whether fresh GPU
   recomputation has also passed.
3. All 800 historical validation frames overlap the published Testing archive.
   The archive's two trajectories cannot be relabeled independent simply by
   removing those individual frames. Entire excluded trajectories are required;
   the newer, different-composition transferability trajectories must be treated
   separately from this historical benchmark.
4. Missing graph caches prevent exact historical-label reconstruction. A
   40-frame/subset diagnostic gives raw energy errors around 5.4 eV/atom but
   reproduces small numerical errors with the suspected double normalization.
   This is strong diagnostic evidence, not proof of the missing cache contents.
   Committee fits are **not authorized by the audit gate**. No production model
   has been corrected/retrained silently.
5. Cleavage occurred near a heating target of 338 K. Original hydrogen-type mass
   was 3.016049 amu in annealing and 1.00783 amu in the final NPT/NVT stages.
   Static cluster energy rises and heating waiting times are not 300 K barriers
   or rate constants.

## Replica protocol

Four independent Packmol seeds and velocity/thermostat seeds; each box has 20
LiTFSI, 96 DME, 143 DOL and two initial Li2S6 units (3445 atoms). Packing and
loose variable-cell FIRE preparation reproduce the archived input workflow.

| Stage | End step | dt (fs) | Temperature (K) | Duration (ps) | Ensemble |
|---|---:|---:|---|---:|---|
| Initial warmup | 100000 | 2 | 50 to 300 | 200 | Langevin + NPH |
| Heat | 300000 | 1 | 300 to 350 | 200 | Langevin + NPH |
| Hold | 400000 | 1 | 350 | 100 | Langevin + NPH |
| Cool | 600000 | 1 | 350 to 300 | 200 | Langevin + NPH |
| Density equilibration | 800000 | 0.5 | 300 | 100 | NPT |
| Production | 1400000 | 0.5 | 300 | 300 | NVT |

The inference adapter is opt-in only after energy/force/stress parity on four
periodic states plus a fragmented cluster and a short LAMMPS dynamics check.
It changes graph construction/transfers and derivative graph retention, not
the potential, masses, integrator or timesteps. Each replica freezes evaluator
checksums. Native MatGL export and cuEquivariance are not assumed compatible
with this legacy DGL M3GNet model. CuEquivariance integration is not attempted.

Every attempt has its own dump and restart lineage. Analysis discards a previous
attempt's tail beyond the restart used by its successor. Langevin reseeding on
restart is recorded; bitwise continuation is not promised. Checkpoints are
every 5000 steps; coordinates/velocities every 10 steps. Preempted jobs may
require manual requeue depending on cluster policy. No automatic run extends
past the preparation-time 14-day deadline. Incomplete sampling stays incomplete.

Geometric event thresholds are predeclared in each manifest: initial close S-S
pairs, sustained separation for 1 ps, cutoff sensitivity 2.6/2.8/3.0 Angstrom;
Li departure sensitivity 3.5/4.0/4.5 Angstrom. All electrolyte lithium IDs are
tracked. Two polysulfides in one box are correlated observations, not replicas.

## Bounded DFT and controls

- `prepare_dft.py`: HSE06 fixed final geometry with AFM/FM/nonmagnetic initial
  states; explicit HSE-specific D3 zero damping, original 600 eV/cell/PAW/charge.
  Initial 12-iteration/4-hour cap is a timing test, not a converged label.
  A one-node pilot exceeded memory; a two-node retry failed in MPI before
  convergence. A final bounded single-node retry uses 450 GB and NCORE=30;
  further hybrid expansion remains gated on its measured convergence/cost.
- PS/non-PS 20-step, 300 K NPT timing pilots converged electronically. They
  justified bounded 2 ps extensions, not equilibrium-density claims. ML controls
  match composition, initial geometry, temperature, pressure and isotope masses;
  VASP and LAMMPS barostat algorithms differ and are documented.
- Never publish POTCAR, WAVECAR or CHGCAR from these private working folders.
  HSE final-state spin localization would not establish an original spin-restricted
  MLIP activation barrier. Filippo leads FES and within-state fluctuations.

## Commands

Run CPU checks in an ASE/NumPy/pymatgen/matplotlib environment:

```bash
python -m unittest discover -s revision_2026 -p 'test_revision.py' -v
python revision_2026/collect_status.py
python revision_2026/analyze_replicas.py
python revision_2026/density_audit.py
python revision_2026/analyze_density.py
python revision_2026/analyze_spin.py
```

`analyze_training.py` and `analyze_path.py` require original private data paths.
`reevaluate_path.py` also needs the archived CUDA environment and a GPU.
`build_revision_documents.py` creates a new working copy only, refusing to
overwrite an existing revision. It requires python-docx and pypdf, installed
locally in the work directory rather than altering the historical environment.

Submit through `sbatch --export=ALL,REVISION_CODE=/absolute/path/revision_2026`.
Do not resubmit jobs already listed in `results/status.json`. Read failed
attempt diagnostics before restarting; never hide an exploded trajectory.

## Acceptance and scope

Release artifacts have checksums. Required checks include atom mappings, units,
restart continuity, electronic convergence, original/accelerated numerical
agreement and block uncertainty. The working response tracker covers R1.1-5
and R2.1-4. Neither a pilot nor a prepared input counts as resolving a scientific
reviewer concern. Final figures, claims and page references require the actual
remaining results, including non-events and inconclusive tests.

References: [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/index.html),
[MatGL LAMMPS](https://github.com/materialyzeai/matgl/blob/main/lammps/README.md),
[VASP hybrid settings](https://vasp.at/wiki/List_of_hybrid_functionals),
[D3 functional parameters](https://github.com/dftd3/simple-dftd3/blob/main/assets/parameters.toml).
The memory-oriented HSE layout follows the role of [VASP NCORE](https://vasp.at/wiki/NCORE);
it is a resource pilot, not a change to the physical model.
