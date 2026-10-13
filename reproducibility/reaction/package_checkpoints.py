#!/usr/bin/env python3
"""Package selected LAMMPS restarts after a zero-step write_data export."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import numpy as np
from ase.io import read, write

SELECTIONS = [
    (100, 50000, "equilib_npt.restart.50000", "xyz.lammpstrj"),
    (260, 160000, "HU_300-350K_1fs/prev_job/heat_nph.restart.160000",
     "HU_300-350K_1fs/prev_job/heat_300to400.lammpstrj"),
    (330, 230000, "HU_300-350K_1fs/heat_nph.restart.230000",
     "HU_300-350K_1fs/heat_300to400.lammpstrj"),
    (340, 240000, "HU_300-350K_1fs/heat_nph.restart.240000",
     "HU_300-350K_1fs/heat_300to400.lammpstrj"),
]
NUMBERS = {1: 6, 2: 9, 3: 1, 4: 3, 5: 7, 6: 8, 7: 16}


def raw_velocities(path):
    lines = path.read_text().splitlines()
    start = lines.index("Velocities") + 1
    rows = {}
    for line in lines[start:]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Unexpected velocity record")
        rows[int(fields[0])] = [float(x) for x in fields[1:]]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ps-root", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / "snapshots"
    output.mkdir(exist_ok=True)
    manifest = []
    for time, step, restart, dump in SELECTIONS:
        stem = f"ps_{time}ps"
        data = output / (stem + ".data")
        shutil.copy2(args.exports / f"lis_{time}ps.data", data)
        restart_out = output / (stem + ".restart")
        shutil.copy2(args.ps_root / restart, restart_out)
        atoms = read(data, format="lammps-data", atom_style="atomic", units="metal",
                     Z_of_type=NUMBERS, sort_by_id=True, read_image_flags=False)
        assert len(atoms) == 3445
        assert np.array_equal(atoms.arrays["id"], np.arange(1, 3446))
        assert atoms.has("momenta")
        assert np.isclose(atoms.get_masses()[atoms.numbers == 1][0], 3.016049)
        atoms.info.update(physical_time_ps=time, original_step=step,
                          velocities_source="original binary restart")
        write(output / (stem + ".extxyz"), atoms)
        velocities = raw_velocities(data)
        assert len(velocities) == len(atoms)
        with (output / (stem + "_velocities.csv")).open("w") as f:
            writer = csv.writer(f)
            writer.writerow(["lammps_id", "vx_A_per_ps", "vy_A_per_ps", "vz_A_per_ps"])
            writer.writerows([i, *velocities[i]] for i in sorted(velocities))
        ss = float(atoms.get_distance(3431, 3432, mic=True))
        li = {str(i): min(float(atoms.get_distance(i-1, s-1, mic=True))
                         for s in range(3430,3436)) for i in [3436,3437]}
        manifest.append({
            "file_stem": stem, "physical_time_ps": time, "lammps_step": step,
            "source_restart_relative_to_PS": restart,
            "source_dump_relative_to_PS": dump,
            "atoms": len(atoms), "cell_A": atoms.cell.array.tolist(),
            "temperature_from_3N_dof_K": atoms.get_temperature(),
            "central_SS_A": ss, "original_Li_min_S_distances_A": li,
            "target_temperature_K": {100:175,260:315,330:332.5,340:335}[time],
            "velocity_units_data_and_csv": "Angstrom/ps",
            "extxyz_velocities": "ASE momenta with isotope masses; use atoms.get_velocities()",
        })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = {f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(output.iterdir()) if f.is_file() and f.name != "checksums.json"}
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
