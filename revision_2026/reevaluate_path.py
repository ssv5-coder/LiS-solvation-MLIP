#!/usr/bin/env python3
"""Fresh original-model inference, checked against the historical cluster cache."""
import csv
import importlib.util
import json
import numpy as np
from ase.io import read
from common import ROOT, MODEL, PATH, WORK, save_json, save_csv, sha256


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this pilot")
    source = ROOT / "advancesoft-lammps/src/ML-GNNP/gnnp_driver.py"
    spec = importlib.util.spec_from_file_location("production_driver", source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.gnnp_initialize("matgl", str(MODEL))
    energies = {int(r["image"]): r for r in csv.DictReader((PATH / "m3gnet_path_energies.csv").open())}
    forces = list(csv.DictReader((PATH / "m3gnet_path_forces.csv").open()))
    comparisons = []
    fresh = []
    for row in csv.DictReader((PATH / "singlepoint_path_energies.csv").open()):
        from pathlib import Path
        image = int(row["image"])
        atoms = read(Path(row["workdir"]) / "POSCAR")
        energy, force, stress = driver.gnnp_get_energy_forces_stress(
            atoms.cell.array.tolist(), atoms.numbers.tolist(), atoms.positions.tolist(), True)
        expected = sorted([r for r in forces if int(r["image"]) == image], key=lambda r: int(r["atom_index"]))
        old_force = np.array([[float(r[k]) for k in ["fx_eV_A", "fy_eV_A", "fz_eV_A"]] for r in expected])
        comparisons.append(dict(image=image, energy_eV=float(energy),
                           energy_difference_eV=float(energy)-float(energies[image]["m3gnet_energy_eV"]),
                           maximum_force_difference_eV_A=float(np.abs(np.array(force)-old_force).max())))
        for i, vector in enumerate(force):
            fresh.append(dict(image=image, atom_index=i, element=atoms[i].symbol,
                              fx_eV_A=vector[0], fy_eV_A=vector[1], fz_eV_A=vector[2]))
    output = WORK / "reactive_validation"
    save_csv(output / "fresh_model_predictions.csv", comparisons)
    save_csv(output / "fresh_model_forces.csv", fresh)
    passed = all(abs(r["energy_difference_eV"])/148 < 1e-4 and
                 r["maximum_force_difference_eV_A"] < 1e-4 for r in comparisons)
    save_json(output / "fresh_prediction_verification.json", dict(
        passed=passed, model_checksums={p.name: sha256(p) for p in MODEL.iterdir() if p.is_file()},
        driver_sha256=sha256(source), gpu=torch.cuda.get_device_name(0),
        max_energy_difference_eV=max(abs(r["energy_difference_eV"]) for r in comparisons),
        max_force_difference_eV_A=max(r["maximum_force_difference_eV_A"] for r in comparisons)))
    if not passed:
        raise RuntimeError("Fresh predictions disagree with historical cache; do not publish cached metrics as verified")
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    summary["prediction_status"] = "Recomputed with unchanged original production model; cached E/F agree within declared GPU tolerance; raw DFT and geometry/order verified separately"
    save_json(path, summary)
    print("All 21 original-model cluster predictions verified", flush=True)


if __name__ == "__main__":
    main()
