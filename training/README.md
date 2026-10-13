# Fine-tuning and historical provenance

The archived final-stage script is
[original/Finetuning_smaller_ada.py](original/Finetuning_smaller_ada.py).
It saves `big_boy_m3gnet_smaller_stress_ada_2_best`, whose three serialized
files have the same SHA-256 checksums as the released production potential.
The associated version-53 metrics reproduce the reported final training and
validation errors. They are included with the original hyperparameters.

## Full fine-tuning

The script passes the complete loaded M3GNet network to
`PotentialLightningModule`; there are no layer-freezing instructions.
MatGL's default Adam optimizer receives the module parameters.
The portable script explicitly enables gradients on every network parameter
and records trainable and frozen parameter counts.
Element-reference offsets are inherited from the initialization potential.

Verified final-stage settings:

| Setting | Value |
|---|---|
| Initialization | Included `models/final_stage_initialization` |
| Network / three-body cutoffs | 5.0 / 4.0 Angstrom |
| Epochs | 20 (0 through 19 in the saved log) |
| Learning rate | 0.0001 |
| Default MatGL scheduler | CosineAnnealingLR, T_max=50, eta_min=0.00001 |
| Energy / force loss weights | 1.0 / 1.0 |
| Stress loss weight | 0.01 to 0.1 over epochs 0 through 19 |
| Batch size / accumulation | 28 / 2 |
| Gradient clipping | 1.0 |
| Torch / dataset shuffle seeds | 0 / 42 |
| Loss | MSE |

This is the final stage starting from an already adapted model, not a claim
that the entire prior training history comprised 20 epochs. The submitted
600-epoch description and stress weight ending at 1.0 are not verified by
these final-stage files.

## Run

Install the repository requirements plus this directory's requirements.
Supply separate labeled ASE trajectories for fitting and validation:

```bash
python training/finetune.py \
  --train /path/to/train.traj \
  --validation /path/to/validation.traj \
  --output /path/to/new_training_run --accelerator gpu
```

For a one-batch installation check, add `--smoke-test --workers 0` and provide
small input trajectories. This does not export a fitted model.
Input labels are total energies in eV, forces in eV/Angstrom, and ASE-convention
stresses in eV/Angstrom^3. Stresses are converted to GPa for MatGL.
Training and validation graphs are cached in separate output directories.
The default initialization is the bundled historical final-stage input model.

To reconstruct the historical input selection from the already published
archives, extract them into a directory containing `Training/` and `Testing/`,
then run:

```bash
python training/assemble_historical_inputs.py \
  --data-root /path/to/extracted_archives --output /path/to/new_historical_inputs
```

This selection deliberately reconstructs the recovered historical overlap.
It is for auditing, not a recommended split for new model development.

## Reconstruction limits requiring attention

The historical script loads precomputed `save_train_best` and
`save_val_best` graph caches, which are no longer present at those paths.
The original preprocessing notebook is included with outputs removed.
That notebook divides energy labels by atom count, whereas MatGL 1.2.3's
loss also divides energy targets by atom count. It is not established that
this notebook version generated the actual caches. The portable script uses
the documented MatGL total-energy convention and should not be described as
bitwise reproduction of the missing caches.
The explicit `--energy-label-convention historical-per-atom` option is
available to investigate the archived notebook convention. It causes a second
normalization inside the MatGL loss and is not the default for new training.

The historical `train.traj` has 120,000 frames; checks of block boundaries
match alternating nonLiPS/LiPS boxes 1-10 in the published Training archive.
The historical validation `test.traj` contains 800 frames. A geometry
audit against the published Testing archive is provided in
`historical_validation_matches.json` (positions and cell rounded to 1e-8 A).
All 800 frames match: 600 PS and 200 non-PS frames. Matching frames must be excluded
from claims of independent held-out testing; preferably hold out entire
trajectories to avoid temporal leakage. Do not automatically use the Testing
archive as validation for a new training run.

For clarity, `finetune.py` is a portable reconstruction with explicit input
paths and standard label units; `original/` preserves the recovered evidence.
No new production model is trained by this release.

## References

- Chen and Ong, M3GNet, *Nature Computational Science* (2022),
  DOI: [10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3).
- Ko et al., MatGL, *npj Computational Materials* (2025),
  DOI: [10.1038/s41524-025-01742-y](https://doi.org/10.1038/s41524-025-01742-y).
- [MatGL source](https://github.com/materialsvirtuallab/matgl):
  see `PotentialLightningModule` and `configure_optimizers` in the installed
  version's `matgl/utils/training.py`.
