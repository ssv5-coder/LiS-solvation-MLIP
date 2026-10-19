#!/usr/bin/env python3
"""Reproduce the original loose, variable-cell FIRE preparation on a new packing."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import read, write
from common import MODEL, ELEMENTS, WORK, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if (folder / "initial.data").exists():
        raise FileExistsError("Initial structure exists; refusing to re-relax")
    import torch
    import matgl
    from matgl.ext.ase import Relaxer
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; refusing CPU fallback")
    torch.set_default_device("cuda")
    if (WORK / "gpu_optimization/selected_driver.json").exists():
        from fast_graph import install
        install(inference_grad=True)
    potential = matgl.load_model(str(MODEL)).to("cuda")
    atoms = read(folder / "packed.pdb")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(atoms) == 3445
    assert atoms.get_chemical_symbols()[-16:] == (["S"]*6 + ["Li"]*2)*2
    atoms.set_cell([manifest["initial_cell_A"]]*3)
    atoms.pbc = True
    mass = atoms.get_masses()
    mass[atoms.numbers == 1] = 2.01410177812
    atoms.set_masses(mass)
    relaxer = Relaxer(potential=potential, relax_cell=True)
    result = relaxer.relax(atoms, fmax=2.0, steps=500, traj_file=str(folder / "relaxation.pkl"), verbose=True)
    maxforce = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    if not np.isfinite(atoms.positions).all() or not np.isfinite(maxforce):
        raise RuntimeError("Nonfinite relaxed packing")
    if maxforce > 2.0:
        raise RuntimeError(f"FIRE did not reach the original loose force criterion: {maxforce}")
    if not 0.8 < atoms.get_volume() / manifest["initial_cell_A"]**3 < 1.2:
        raise RuntimeError("Large preparation cell change requires inspection")
    # Original LAMMPS input overrides the H mass to this value at initialization.
    mass[atoms.numbers == 1] = 3.016049
    atoms.set_masses(mass)
    write(folder / "initial.data", atoms, format="lammps-data", atom_style="atomic", specorder=ELEMENTS, masses=True)
    atoms.calc = None
    write(folder / "initial.extxyz", atoms)
    save_json(folder / "preparation.json", dict(natoms=len(atoms), max_atomic_force_eV_A=maxforce,
              gpu=torch.cuda.get_device_name(0),
              volume_A3=atoms.get_volume(), data_sha256=sha256(folder / "initial.data"),
              model=str(MODEL), packing_sha256=sha256(folder / "packed.pdb"),
              note="Same MatGL variable-cell Relaxer and fmax=2 eV/A, 500-step cap as original notebook"))


if __name__ == "__main__":
    main()
