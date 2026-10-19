#!/usr/bin/env python3
"""Launch one bounded static-HSE or density pilot and preserve convergence status."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
from common import save_json, DFT_MASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if (folder / "OUTCAR").exists():
        raise FileExistsError("Existing OUTCAR: prepare a new continuation directory instead of overwriting")
    metadata = json.loads((folder / "provenance.json").read_text())
    seed = metadata.get("wavecar_seed")
    if seed:
        # Copy-on-write when supported; never expose the source WAVECAR to VASP writes.
        subprocess.run(["cp", "--reflink=auto", seed, str(folder / "WAVECAR")], check=True)
        incar = Incar.from_file(folder / "INCAR")
        # This pymatgen version reads POMASS as a scalar. Restore the full seven-type list.
        incar.update(dict(ISTART=1, ICHARG=0, POMASS=DFT_MASSES))
        incar.write_file(folder / "INCAR")
    start = time.time()
    exe = "/sdf/group/suncat/sw/electronicstructure/vasp-6.4.2/bin/vasp_std"
    with (folder / "vasp_stdout.log").open("w") as stdout, (folder / "vasp_stderr.log").open("w") as stderr:
        result = subprocess.run(["mpirun", "-n", os.environ["SLURM_NTASKS"], exe], cwd=folder, stdout=stdout, stderr=stderr)
    text = (folder / "OUTCAR").read_text(errors="replace") if (folder / "OUTCAR").exists() else ""
    save_json(folder / "pilot_status.json", dict(returncode=result.returncode, wall_seconds=time.time()-start,
              ediff_reached_count=text.count("aborting loop because EDIFF is reached"),
              final_timing_present="General timing and accounting" in text,
              note="A completed timing pilot is not proof of SCF convergence or density equilibration"))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
