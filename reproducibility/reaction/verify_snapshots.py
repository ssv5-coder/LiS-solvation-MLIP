#!/usr/bin/env python3
"""Check snapshot IDs, velocity conversion and optional source-dump agreement."""
from pathlib import Path
import argparse
import csv
import json
import hashlib
import itertools
import numpy as np
from ase.io import read
from ase.calculators.lammps.unitconvert import convert


def source_frames(path, steps):
    remaining = set(steps)
    with path.open() as f:
        while remaining:
            if f.readline().strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Missing frame in {path}: {remaining}")
            step = int(f.readline())
            f.readline()
            n = int(f.readline())
            f.readline()
            bounds = np.array([list(map(float,f.readline().split())) for _ in range(3)])
            columns = f.readline().split()[2:]
            rows = list(itertools.islice(f,n))
            if step in remaining:
                positions = np.zeros((n,3))
                for row in rows:
                    values = row.split()
                    positions[int(values[columns.index("id")])-1] = [float(values[columns.index(c)]) for c in ("x","y","z")]
                yield step, positions, bounds[:,1]-bounds[:,0]
                remaining.remove(step)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ps-root",type=Path,help="Optional original trajectory directory")
    args=parser.parse_args()
    folder=Path(__file__).resolve().parent/"snapshots"
    manifest=json.loads((folder/"manifest.json").read_text())
    checksums=json.loads((folder/"checksums.json").read_text())
    for name,expected in checksums.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected, name
    results=[]
    atoms_by_step={}
    for entry in manifest:
        stem=entry["file_stem"]
        atoms=read(folder/(stem+".extxyz"))
        ids=atoms.arrays["id"]
        assert len(atoms)==3445 and np.array_equal(ids,np.arange(1,3446))
        with (folder/(stem+"_velocities.csv")).open() as f:
            rows=list(csv.DictReader(f))
        vel=np.array([[float(r[k]) for k in ("vx_A_per_ps","vy_A_per_ps","vz_A_per_ps")] for r in rows])
        error=float(np.max(np.abs(atoms.get_velocities()-convert(vel,"velocity","metal","ASE"))))
        assert error<1e-8, error
        atoms_by_step[entry["lammps_step"]]=atoms
        results.append({"time_ps":entry["physical_time_ps"],"max_velocity_roundtrip_error_ASE":error})
    if args.ps_root:
        sources={}
        for e in manifest:
            sources.setdefault(e["source_dump_relative_to_PS"],[]).append(e["lammps_step"])
        for source,steps in sources.items():
            for step,positions,lengths in source_frames(args.ps_root/source,steps):
                atoms=atoms_by_step[step]
                delta=atoms.positions-positions
                delta-=atoms.cell.lengths()*np.rint(delta/atoms.cell.lengths())
                error=float(np.max(np.abs(delta)))
                assert error<0.0001, (source,step,error)
                assert np.max(np.abs(atoms.cell.lengths()-lengths))<1e-8
                index=next(i for i,e in enumerate(manifest) if e["lammps_step"]==step)
                results[index].update(source_dump_max_coordinate_difference_A=error,
                                      source_dump_match=True)
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
