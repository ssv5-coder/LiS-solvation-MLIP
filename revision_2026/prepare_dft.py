#!/usr/bin/env python3
"""Prepare bounded HSE and small-cell density pilots, never modify source runs."""
import argparse
import shutil
from pathlib import Path
from ase.io import read, write
from pymatgen.io.vasp.inputs import Incar
from common import ROOT, WORK, SPIN, DFT_MASSES, sha256, save_json


def fresh(folder):
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite {folder}")
    folder.mkdir(parents=True)


def source_inputs(folder, source, structure="POSCAR"):
    shutil.copy2(source / structure, folder / "POSCAR")
    shutil.copy2(source / "KPOINTS", folder / "KPOINTS")
    # Licensed PAW datasets stay local, outside GitHub.
    (folder / "POTCAR").symlink_to((source / "POTCAR").resolve())


def prepare_hse(pilot_name="pilot", ncore=10):
    source = SPIN / "stage2"
    atoms = read(source / "POSCAR")
    assert len(atoms) == 148
    assert atoms.get_chemical_symbols()[-6:] == ["S"] * 6
    for state in ["afm", "fm", "nonmagnetic"]:
        folder = WORK / "hse" / state / pilot_name
        fresh(folder)
        source_inputs(folder, source)
        incar = Incar.from_file(source / "INCAR")
        incar.pop("NPAR", None)
        incar.update(dict(SYSTEM=f"Frame F HSE06 {state} static pilot", LHFCALC=True,
                         HFSCREEN=0.2, AEXX=0.25, ALGO="Damped", TIME=0.4,
                         NSW=0, IBRION=-1, ISIF=0, ISPIN=2 if state != "nonmagnetic" else 1,
                         NELM=12, NCORE=ncore, ISYM=0, ISTART=0, ICHARG=2,
                         LCHARG=True, LWAVE=True, LORBIT=11,
                         VDW_SR=1.129, VDW_S8=0.109, POMASS=DFT_MASSES))
        if state == "nonmagnetic":
            incar.pop("MAGMOM", None)
        else:
            incar["MAGMOM"] = [0.0] * 142 + [1.0] * 3 + ([-1.0] * 3 if state == "afm" else [1.0] * 3)
        incar.write_file(folder / "INCAR")
        final = Incar(incar)
        final.update(dict(NELM=240, ISTART=1, ICHARG=0))
        final.write_file(folder.parent / "INCAR.production")
        # Existing AFM PBE wavefunctions can initialize this pilot without altering originals.
        save_json(folder / "provenance.json", dict(source=str(source), natoms=len(atoms),
                  poscar_sha256=sha256(folder / "POSCAR"), potcar_sha256=sha256(folder / "POTCAR"),
                  wavecar_seed=str(SPIN / "WAVECAR.spin") if state == "afm" else None,
                  dispersion="D3 zero damping, explicit HSE06 rs6=1.129 and s8=0.109; VASP default s6=1",
                  dispersion_reference="https://github.com/dftd3/simple-dftd3/blob/main/assets/parameters.toml",
                  hse_reference="https://vasp.at/wiki/List_of_hybrid_functionals",
                  pilot="12 SCF iterations, at most 4 wall-hours; not a converged result unless EDIFF is actually reached"))


def prepare_density(pilot_name="pilot"):
    sources = {
        "PS": ROOT / "AIMD/DME_DOL_1MLiTFSI_1MLi2S6/box1/04_NVT_300",
        "nonPS": ROOT / "AIMD/DME_DOL_1MLiTFSI/NVT1000RandomBoxes/box1/04_NVT_300",
    }
    for chemistry, source in sources.items():
        folder = WORK / "density" / chemistry / pilot_name
        fresh(folder)
        source_inputs(folder, source, "CONTCAR")
        atoms = read(folder / "POSCAR")
        # A fixed-cell NVT CONTCAR has ionic but no compatible lattice velocities.
        # Start both NPT degrees of freedom afresh at 300 K, keeping geometry fixed.
        atoms.arrays.pop("momenta", None)
        write(folder / "POSCAR", atoms, format="vasp", direct=True, vasp5=True)
        incar = Incar.from_file(source / "INCAR")
        incar.pop("SMASS", None)
        incar.pop("NPAR", None)
        incar.update(dict(SYSTEM=f"{chemistry} 300K NPT density timing pilot",
                         ALGO="Normal", ISTART=0, ICHARG=2, IBRION=0,
                         NSW=20, POTIM=1.0, TEBEG=300.0, TEEND=300.0,
                         MDALGO=3, ISIF=3, PSTRESS=0.00101325,
                         LANGEVIN_GAMMA=[10.0]*7, LANGEVIN_GAMMA_L=10.0,
                         PMASS=1000.0, RANDOM_SEED=[8371 if chemistry == "PS" else 9371, 0, 0],
                         NCORE=8, LCHARG=True, LWAVE=True, POMASS=DFT_MASSES))
        incar.write_file(folder / "INCAR")
        save_json(folder / "provenance.json", dict(source=str(source), natoms=len(atoms),
                  volume_A3=atoms.get_volume(), poscar_sha256=sha256(folder / "POSCAR"),
                  potcar_sha256=sha256(folder / "POTCAR"),
                  pilot_duration_ps=0.020, target_pressure_bar=1.01325,
                  initialization="Coordinates/cell from final NVT; fresh 300K ionic/lattice velocities, no incompatible NVT predictor-corrector block",
                  caution="Timing/stability pilot only, NOT an equilibrium density estimate. All seven simulation masses are inherited from AIMD POMASS."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-name", default="pilot")
    parser.add_argument("--only", choices=["all", "hse", "density"], default="all")
    parser.add_argument("--hse-ncore", type=int, default=10)
    args = parser.parse_args()
    if args.only in ["all", "hse"]:
        if args.hse_ncore < 1 or 120 % args.hse_ncore:
            raise ValueError("HSE NCORE must divide the 120-rank resource layout")
        prepare_hse(args.pilot_name, args.hse_ncore)
    if args.only in ["all", "density"]:
        prepare_density(args.pilot_name)
