# Reviewer reproducibility release

## For Filippo

1. See [reaction/README.md](reaction/README.md) for full periodic pre-cleavage
   snapshots, genuine stored velocities, atom IDs and physical times.
2. Use the released production model in `models/fine_tuned_m3gnet_model`.
3. See [lammps/README.md](lammps/README.md) for AdvanceSoft version, source
   commit, patch, Python driver and build settings.
4. `reaction/continue_330ps.in` supplies a continuation example.
   `reaction/monitor.plumed` supplies diagnostic CV definitions without bias.

## For training reproducibility

[training/README.md](../training/README.md) documents full-network fine-tuning,
the verified final-stage hyperparameters, a portable script, original
script/notebook/logs, and the bundled final-stage initialization model.

The source audit found details that need reconciliation in the revision:

- Final-stage logs show 20 epochs and stress weight 0.01 to 0.1.
- The original preprocessing notebook's energy normalization differs from
  the total-energy convention of the installed MatGL loss. Historical caches
  are unavailable, so the exact executed preprocessing remains unresolved.
- All 800 historical validation structures match geometries in the published
  Testing archive. They are identified individually for a split audit.
- The observed cleavage belongs to the heating trajectory; the ramp segment
  is 325-350 K, with a 1 fs timestep and hydrogen-type mass 3.016049 amu.

These items are reported from files and logs, not inferred from manuscript
descriptions. They should be resolved before claiming exact retraining or
fully independent validation. The existing production model is unchanged.

This release supplies selected states and code. It does not establish a
free-energy barrier, convergence of enhanced sampling, or DFT confirmation of
the trajectory's kinetics. New replicas, HSE calculations, AIMD S-S statistics
and a DFT/ML-MD density comparison remain separate revision tasks.
