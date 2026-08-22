#!/usr/bin/env python3
"""Evaluate an ASE-readable structure with the released M3GNet potential."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from matgl.apps.pes import Potential
from matgl.ext.ase import M3GNetCalculator


REPOSITORY_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPOSITORY_ROOT / "models" / "fine_tuned_m3gnet_model"
GPA_TO_EV_PER_ANGSTROM3 = 1.0 / 160.21766208


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ASE-readable structure with the Li-S M3GNet model."
    )
    parser.add_argument(
        "structure",
        help="Input structure, for example input.extxyz, POSCAR, or input.traj",
    )
    args = parser.parse_args()

    structure_path = Path(args.structure)
    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure not found: {structure_path}")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")

    atoms = read(structure_path)
    potential = Potential.load(MODEL_PATH)
    atoms.calc = M3GNetCalculator(
        potential=potential,
        stress_weight=GPA_TO_EV_PER_ANGSTROM3,
    )

    energy = atoms.get_potential_energy()
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(), dtype=float)

    print(f"Structure : {structure_path}")
    print(f"Atoms     : {len(atoms)}")
    print(f"Energy    : {energy:.10f} eV")
    print("\nForces (eV/Angstrom)")
    print(forces)
    print("\nStress (eV/Angstrom^3; ASE Voigt order xx yy zz yz xz xy)")
    print(stress)


if __name__ == "__main__":
    main()
