"""Compare both historical label interpretations; do not rescale forces or refit."""
import numpy as np
from ase.io.trajectory import Trajectory
from common import TRAINING, WORK, save_csv, save_json


def audit(driver):
    rows = []
    for name in ["train", "test"]:
        with Trajectory(TRAINING / f"{name}.traj") as frames:
            for index in np.linspace(0, len(frames)-1, 40, dtype=int):
                atoms = frames[int(index)]
                predicted, forces, stress = driver.gnnp_get_energy_forces_stress(atoms.cell.array.tolist(), atoms.numbers.tolist(), atoms.positions.tolist(), True)
                reference = atoms.get_potential_energy()
                n = len(atoms)
                rows.append(dict(subset=name, frame=int(index), natoms=n,
                            dft_total_eV=float(reference), model_total_eV=float(predicted),
                            standard_energy_error_meV_atom=1000*(float(predicted)-reference)/n,
                            historical_double_normalized_error_meV=1000*(float(predicted)-reference/n)/n,
                            force_mae_meV_A=float(np.mean(np.abs(np.asarray(forces)-atoms.get_forces())))*1000))
    save_csv(WORK / "training_audit/energy_label_interpretations.csv", rows)
    summary = []
    for name in ["train", "test"]:
        subset = [r for r in rows if r["subset"] == name]
        summary.append(dict(subset=name, nframes=len(subset),
                       standard_energy_mae_meV_atom=float(np.mean([abs(r["standard_energy_error_meV_atom"]) for r in subset])),
                       double_normalized_energy_mae=float(np.mean([abs(r["historical_double_normalized_error_meV"]) for r in subset]))))
    save_json(WORK / "training_audit/normalization_summary.json", dict(results=summary,
              caveat="Diagnostic sample, not the reported full benchmark. Agreement with one convention does not recover missing graph caches. No production-model change."))
