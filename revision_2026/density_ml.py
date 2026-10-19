#!/usr/bin/env python3
"""Short matched-composition ML NPT controls; not a claim of equilibrated density."""
import argparse
import subprocess
import numpy as np
from ase.io import read, write
from common import ROOT, MODEL, WORK, ELEMENTS, DFT_MASSES, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chemistry", choices=["PS", "nonPS"])
    args = p.parse_args()
    folder = WORK / "density" / args.chemistry / "ml_pilot"
    if (folder / "log.lammps").exists():
        raise FileExistsError("Existing control; do not overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    atoms = read(WORK / "density" / args.chemistry / "retry02/POSCAR")
    mass_map = dict(zip(ELEMENTS, DFT_MASSES))
    atoms.set_masses([mass_map[s] for s in atoms.get_chemical_symbols()])
    has_velocities = atoms.has("momenta") and np.linalg.norm(atoms.get_velocities()) > 0
    write(folder / "initial.data", atoms, format="lammps-data", atom_style="atomic",
          masses=True, velocities=has_velocities, specorder=ELEMENTS, force_skew=True)
    velocity = "" if has_velocities else "velocity all create 300.0 93719 dist gaussian\n"
    text = f'''units metal
boundary p p p
atom_style atomic
read_data initial.data
pair_style gnnp/gpu {ROOT}/advancesoft-lammps/src/ML-GNNP
pair_coeff * * matgl {MODEL} C F H Li N O S
{velocity}timestep 0.001
fix 1 all langevin 300.0 300.0 0.1 77419
fix 2 all nph tri 1.01325 1.01325 2.0
thermo 10
thermo_style custom step time pe temp press vol density
thermo_modify flush yes lost error
dump frames all custom 20 trajectory.lammpstrj id type x y z vx vy vz
dump_modify frames sort id
restart 500 checkpoint.*.restart
run 2000
write_restart final.restart
write_data final.data
'''
    (folder / "input.lammps").write_text(text)
    save_json(folder / "provenance.json", dict(natoms=len(atoms), duration_ps=2.0,
               masses=mass_map, velocities_inherited=bool(has_velocities),
               comparison="Matched composition, initial geometry, temperature, pressure and masses; different MD integrator/barostat from VASP. Timing/stability pilot, not equilibrium density."))
    subprocess.run([str(ROOT / "advancesoft-lammps/build/lmp"), "-in", "input.lammps"], cwd=folder, check=True)


if __name__ == "__main__":
    main()
