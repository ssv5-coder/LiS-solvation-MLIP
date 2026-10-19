#!/usr/bin/env python3
"""Expand only an electronically converged timing pilot to a bounded 2 ps run."""
import argparse
import json
import shutil
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
from common import WORK, DFT_MASSES, save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chemistry", choices=["PS", "nonPS"])
    args = p.parse_args()
    source = WORK / "density" / args.chemistry / "retry02"
    status = json.loads((source / "pilot_status.json").read_text())
    if status["returncode"] != 0 or status["ediff_reached_count"] != 20 or not status["final_timing_present"]:
        raise RuntimeError("Pilot did not complete 20 converged ionic steps")
    estimate_hours = status["wall_seconds"] * 100 / 3600
    if estimate_hours > 20:
        raise RuntimeError("Projected 2 ps cost exceeds the 20-hour expansion gate")
    folder = source.parent / "npt_2ps"
    folder.mkdir()
    shutil.copy2(source / "CONTCAR", folder / "POSCAR")
    shutil.copy2(source / "KPOINTS", folder / "KPOINTS")
    (folder / "POTCAR").symlink_to((source / "POTCAR").resolve())
    incar = Incar.from_file(source / "INCAR")
    incar.update(dict(NSW=2000, ISTART=1, ICHARG=0, POMASS=DFT_MASSES))
    incar.write_file(folder / "INCAR")
    save_json(folder / "provenance.json", dict(source=str(source), wavecar_seed=str(source / "WAVECAR"),
              projected_hours_from_pilot=estimate_hours, duration_ps=2.0,
              poscar_sha256=sha256(folder / "POSCAR"),
              warning="No equilibrium-density claim until stationarity and block uncertainties are assessed"))
    print(folder)


if __name__ == "__main__":
    main()
