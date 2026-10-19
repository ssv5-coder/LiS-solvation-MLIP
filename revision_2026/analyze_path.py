#!/usr/bin/env python3
"""Validate cached original-model predictions against raw converged DFT labels."""
import csv
import re
from pathlib import Path
import numpy as np
from ase.io import read
from common import PATH, WORK, save_json, save_csv, sha256


def errors(values):
    values = np.asarray(values)
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values ** 2)))


def main():
    output = WORK / "reactive_validation"
    predictions = {int(r["image"]): r for r in csv.DictReader((PATH / "m3gnet_path_energies.csv").open())}
    force_rows = list(csv.DictReader((PATH / "m3gnet_path_forces.csv").open()))
    order = list(csv.DictReader((PATH / "atom_order.csv").open()))
    original_ids = np.array([int(r["original_atom_id"]) for r in order])
    rows, element_rows, all_errors = [], [], []
    for label in csv.DictReader((PATH / "singlepoint_path_energies.csv").open()):
        image = int(label["image"])
        if label["electronic_converged"] != "True":
            raise RuntimeError(f"Unconverged DFT image {image}")
        outcar = Path(label["energy_source"])
        raw = outcar.read_text(errors="replace")
        if "aborting loop because EDIFF is reached" not in raw:
            raise RuntimeError(f"SCF convergence not found: {outcar}")
        atoms = read(outcar.parent / "vasprun.xml", index=-1)
        original = read(Path(label["workdir"]) / "POSCAR")
        if not np.array_equal(atoms.numbers, original.numbers) or not np.allclose(atoms.positions, original.positions, atol=1e-5, rtol=0) or not np.allclose(atoms.cell, original.cell, atol=1e-6, rtol=0):
            raise RuntimeError(f"DFT geometry/order mismatch for image {image}")
        predicted = sorted([r for r in force_rows if int(r["image"]) == image], key=lambda r: int(r["atom_index"]))
        assert [int(r["atom_index"]) for r in predicted] == list(range(len(atoms)))
        assert [r["element"] for r in predicted] == atoms.get_chemical_symbols()
        force = np.array([[float(r[k]) for k in ("fx_eV_A", "fy_eV_A", "fz_eV_A")] for r in predicted])
        delta = force - atoms.get_forces()
        all_errors.append(delta)
        dft_energy = float(re.findall(r"energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)", raw)[-1])
        if abs(dft_energy - float(label["energy_eV"])) > 1e-6:
            raise RuntimeError("Cached energy disagrees with OUTCAR")
        ml_energy = float(predictions[image]["m3gnet_energy_eV"])
        if not rows:
            dft_reference, ml_reference = dft_energy, ml_energy
        force_mae, force_rmse = errors(delta * 1000)
        central = np.isin(original_ids, [3432, 3433])
        central_mae, central_rmse = errors(delta[central] * 1000)
        row = dict(image=image, time_ps=float(label["time_label"].split()[0]), natoms=len(atoms),
                   central_ss_A=float(label["central_ss_A"]), dft_energy_eV=dft_energy, mlip_energy_eV=ml_energy,
                   energy_error_meV_atom=(ml_energy - dft_energy) * 1000 / len(atoms),
                   relative_energy_error_eV=(ml_energy - ml_reference) - (dft_energy - dft_reference),
                   force_mae_meV_A=force_mae, force_rmse_meV_A=force_rmse,
                   breaking_S_force_mae_meV_A=central_mae, breaking_S_force_rmse_meV_A=central_rmse,
                   dft_outcar=str(outcar), dft_outcar_sha256=sha256(outcar))
        rows.append(row)
        for element in sorted(set(atoms.get_chemical_symbols())):
            mask = np.array(atoms.get_chemical_symbols()) == element
            mae, rmse = errors(delta[mask] * 1000)
            element_rows.append(dict(image=image, element=element, natoms=int(mask.sum()), force_mae_meV_A=mae, force_rmse_meV_A=rmse))
    save_csv(output / "original_model_reactive_errors.csv", rows)
    save_csv(output / "per_element_reactive_errors.csv", element_rows)
    e_mae, e_rmse = errors([r["energy_error_meV_atom"] for r in rows])
    f_mae, f_rmse = errors(np.concatenate(all_errors) * 1000)
    rel_mae, rel_rmse = errors([r["relative_energy_error_eV"] for r in rows])
    save_json(output / "summary.json", dict(nframes=len(rows), energy_mae_meV_atom=e_mae,
              energy_rmse_meV_atom=e_rmse, force_mae_meV_A=f_mae, force_rmse_meV_A=f_rmse,
              relative_energy_mae_eV=rel_mae, relative_energy_rmse_eV=rel_rmse,
              energy_reference_image=rows[0]["image"], geometry="Unrelaxed fixed-composition 148-atom extracted clusters in vacuum; periodic large cell",
              model="Original production model, not path-adapted", prediction_status="Historical predictions cross-checked against atom order and raw DFT; new GPU recomputation pending",
              limitations="These are not bulk reactive DFT labels, free energies, NEB barriers, or independent tests of a path-adapted model."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    t = [r["time_ps"] for r in rows]
    for ax, key, ylabel in zip(axes.flat,
            ["energy_error_meV_atom", "relative_energy_error_eV", "force_mae_meV_A", "breaking_S_force_mae_meV_A"],
            ["Signed energy error (meV/atom)", "Relative-energy error (eV)", "Force MAE (meV/Angstrom)", "Breaking S-pair force MAE (meV/Angstrom)"]):
        ax.plot(t, [r[key] for r in rows], "o-", ms=3, color="#2678b2")
        ax.set(xlabel="Original trajectory time (ps)", ylabel=ylabel)
    fig.savefig(output / "reactive_error_audit.png", dpi=400)
    fig.savefig(output / "reactive_error_audit.pdf")
    print("Path audit:", e_mae, "meV/atom;", f_mae, "meV/Angstrom", flush=True)


if __name__ == "__main__":
    main()
