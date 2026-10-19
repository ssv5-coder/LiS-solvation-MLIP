#!/usr/bin/env python3
"""Report converged static spin tests; missing HSE results remain explicitly missing."""
import csv
import re
import numpy as np
from ase.io import read
from pymatgen.io.vasp.outputs import Outcar
from common import PATH, SPIN, WORK, save_csv, save_json, sha256


def main():
    order = list(csv.DictReader((PATH / "atom_order.csv").open()))
    ids = np.array([int(r["original_atom_id"]) for r in order])
    sources = [("PBE", "spin_final", SPIN / "stage2")]
    sources += [("HSE06", state, folder) for state in ["afm", "fm", "nonmagnetic"]
                for folder in sorted((WORK / "hse" / state).glob("*")) if folder.is_dir()]
    rows, missing = [], []
    reference = read(SPIN / "stage2/POSCAR")
    for functional, state, folder in sources:
        output = folder / "OUTCAR"
        if not output.exists():
            missing.append(dict(state=state, folder=str(folder), reason="no output"))
            continue
        text = output.read_text(errors="replace")
        if "aborting loop because EDIFF is reached" not in text or "General timing and accounting" not in text:
            missing.append(dict(state=state, folder=str(folder), reason="electronic convergence/final timing absent"))
            continue
        atoms = read(folder / "POSCAR")
        if not np.array_equal(reference.numbers, atoms.numbers) or not np.allclose(reference.positions, atoms.positions, atol=1e-7) or not np.allclose(reference.cell, atoms.cell, atol=1e-7):
            raise RuntimeError("Spin comparison requires identical geometry/order/cell")
        parsed = Outcar(output)
        moments = np.array([r["tot"] for r in parsed.magnetization])
        magnetic = re.search(r"ISPIN\s*=\s*(\d+)", text)
        if magnetic and magnetic[1] == "1":
            moments = np.zeros(len(atoms))
        if len(moments) != len(atoms):
            raise RuntimeError("Missing per-ion spin projection")
        energy = float(re.findall(r"energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)", text)[-1])
        for fragment, selected in [("S3_left", [3430,3431,3432]), ("S3_right", [3433,3434,3435])]:
            mask = np.isin(ids, selected)
            if mask.sum() != 3 or set(np.array(atoms.get_chemical_symbols())[mask]) != {"S"}:
                raise RuntimeError("Unexpected original-system sulfur mapping")
            rows.append(dict(functional=functional, initialization=state, fragment=fragment,
                energy_sigma0_eV=energy, projected_fragment_moment_muB=float(moments[mask].sum()),
                projected_absolute_local_moments_muB=float(abs(moments[mask]).sum()),
                source=str(output), outcar_sha256=sha256(output)))
    save_json(WORK / "spin/comparison_status.json", dict(missing=missing,
        limitation="PAW sphere projections omit interstitial spin. Inspect converged CHGCAR spin density and compare only same-functional energies; this does not validate an MLIP reaction barrier."))
    if rows:
        save_csv(WORK / "spin/converged_fragment_moments.csv", rows)
    hse_states = {r["initialization"] for r in rows if r["functional"] == "HSE06"}
    print("Converged HSE states:", sorted(hse_states), "; no hybrid comparison is complete until all requested states are assessed")


if __name__ == "__main__":
    main()
