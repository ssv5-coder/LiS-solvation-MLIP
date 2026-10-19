#!/usr/bin/env python3
"""Inventory actual AIMD ensembles and distinguish imposed from sampled density."""
import re
from pathlib import Path
import numpy as np
from ase.io.trajectory import Trajectory
from ase.data import atomic_masses
from pymatgen.io.vasp.inputs import Incar
from common import ROOT, BASE, WORK, REPO, DFT_MASSES, ELEMENTS, save_csv, save_json


def main():
    inventory, density_rows = [], []
    sources = [("nonPS", ROOT / "AIMD/DME_DOL_1MLiTFSI/NVT1000RandomBoxes"),
               ("PS", ROOT / "AIMD/DME_DOL_1MLiTFSI_1MLi2S6")]
    for chemistry, root in sources:
        for folder in sorted(root.glob("box*/0*")):
            if not (folder / "INCAR").exists():
                continue
            incar = Incar.from_file(folder / "INCAR")
            variable_cell = int(incar.get("ISIF", 2)) == 3
            constant_300 = incar.get("TEBEG") == 300 and incar.get("TEEND") == 300
            inventory.append(dict(chemistry=chemistry, source=str(folder), isif=incar.get("ISIF"),
                             mdalgo=incar.get("MDALGO"), potim_fs=incar.get("POTIM"),
                             tebeg_K=incar.get("TEBEG"), teend_K=incar.get("TEEND"),
                             ispin=incar.get("ISPIN", 1), variable_cell=variable_cell,
                             constant_300K_NPT_candidate=bool(variable_cell and constant_300),
                             note="Input target, not evidence of equilibration"))
            trajectory = folder / "final_with_calculator.traj"
            if not trajectory.exists():
                continue
            with Trajectory(trajectory) as frames:
                sample = sorted(set([0, len(frames)//2, len(frames)-1]))
                for frame in sample:
                    atoms = frames[frame]
                    physical_mass = atomic_masses[atoms.numbers].sum()
                    simulation_mass = sum(dict(zip(ELEMENTS, DFT_MASSES))[s] for s in atoms.get_chemical_symbols())
                    density_rows.append(dict(chemistry=chemistry, source=str(trajectory), frame=frame,
                           natoms=len(atoms), volume_A3=atoms.get_volume(),
                           natural_mass_density_g_cm3=1.66053906660*physical_mass/atoms.get_volume(),
                           dft_mass_density_g_cm3=1.66053906660*simulation_mass/atoms.get_volume(),
                           equilibrium_density_claim=False,
                           classification="NPT temperature ramp / unverified equilibration" if variable_cell else "NVT imposed volume"))
    save_csv(WORK / "density/aimd_ensemble_inventory.csv", inventory)
    save_csv(WORK / "density/existing_aimd_density_samples.csv", density_rows)
    save_json(WORK / "density/inventory_summary.json", dict(ninputs=len(inventory),
              constant_300K_NPT_candidates=sum(r["constant_300K_NPT_candidate"] for r in inventory),
              interpretation="NVT volumes and endpoints of temperature ramps are not equilibrated DFT density predictions.",
              mass_note="Original ML-MD annealing used H=3.016049; original final 300K NPT changes H to 1.00783. AIMD POMASS H=2.0. Matched controls use all seven AIMD masses."))


if __name__ == "__main__":
    main()
