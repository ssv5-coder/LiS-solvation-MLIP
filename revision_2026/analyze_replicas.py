#!/usr/bin/env python3
"""Stream valid restart branches and report geometric events, including censoring."""
import csv
import itertools
import json
import re
from pathlib import Path
import numpy as np
from ase.geometry import find_mic
from common import WORK, physical_time, save_csv


def dump_frames(path):
    """Read complete orthogonal/triclinic custom frames; ignore an interrupted tail."""
    with path.open() as handle:
        while handle.readline():
            try:
                step = int(handle.readline())
                if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError("Unexpected dump header")
                count = int(handle.readline())
                header = handle.readline()
                bounds = np.array([list(map(float, handle.readline().split())) for _ in range(3)])
                lower, upper = bounds[:, 0].copy(), bounds[:, 1].copy()
                cell = np.diag(upper-lower)
                if "xy xz yz" in header:
                    xy, xz, yz = bounds[:, 2]
                    lower[0] -= min(0, xy, xz, xy+xz)
                    upper[0] -= max(0, xy, xz, xy+xz)
                    lower[1] -= min(0, yz)
                    upper[1] -= max(0, yz)
                    cell = np.array([[upper[0]-lower[0], 0, 0], [xy, upper[1]-lower[1], 0], [xz, yz, upper[2]-lower[2]]])
                fields = handle.readline().split()[2:]
                lines = [handle.readline() for _ in range(count)]
                if any(not line for line in lines):
                    return
                data = np.array([list(map(float, line.split())) for line in lines])
                data = data[np.argsort(data[:, fields.index("id")])]
                ids = data[:, fields.index("id")].astype(int)
                if not np.array_equal(ids, np.arange(1, count+1)):
                    raise ValueError("Noncontiguous/reordered atom IDs")
                positions = data[:, [fields.index(x) for x in ["x", "y", "z"]]] - lower
                types = data[:, fields.index("type")].astype(int)
                yield step, cell, positions, types
            except (ValueError, IndexError):
                if not handle.read(1):
                    return
                raise


def valid_frames(folder, stages):
    last = -1
    for stage in stages:
        attempts = sorted((folder / stage["name"]).glob("attempt_*"))
        starts = []
        for attempt in attempts:
            lineage = json.loads((attempt / "lineage.json").read_text())
            match = re.search(r"checkpoint\.(\d+)\.restart", lineage["source_restart"])
            starts.append(int(match.group(1)) if match else stage["start"])
        for index, attempt in enumerate(attempts):
            stop = starts[index+1] if index+1 < len(attempts) else stage["end"]+1
            path = attempt / "trajectory.lammpstrj"
            if not path.exists():
                continue
            for frame in dump_frames(path):
                step = frame[0]
                if starts[index] <= step < stop and step > last:
                    last = step
                    yield frame


def first_persistent(times, values, cutoff, duration=1.0):
    onset = None
    previous = None
    for t, value in zip(times, values):
        if previous is not None and t-previous > 0.1:
            onset = None
        if value > cutoff:
            onset = t if onset is None else onset
            if t-onset >= duration-1e-8:
                return float(onset)
        else:
            onset = None
        previous = t
    return None


def main():
    summaries = []
    for folder in sorted((WORK / "replicas").glob("R0*")):
        manifest = json.loads((folder / "manifest.json").read_text())
        output = folder / "analysis"
        output.mkdir(exist_ok=True)
        measurements = []
        volume_warnings = []
        failed_attempts = list(folder.glob("*/attempt_*/failed.json"))
        with (output / "contacts.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "time_ps", "PS", "central_SS_A", "max_initial_SS_A", "Li1_nearest_S_A", "Li2_nearest_S_A", "associated_Li_ids_within_3p5A"])
            original_pairs = {}
            for step, cell, positions, types in valid_frames(folder, manifest["stages"]):
                ratio = abs(np.linalg.det(cell)) / manifest["initial_cell_A"]**3
                if not 0.5 < ratio < 1.5:
                    volume_warnings.append(step)
                li = np.flatnonzero(types == 4)
                for name in ["LS01", "LS02"]:
                    sulfur = np.array(manifest["atom_ids"][name+"_S"])-1
                    original_li = np.array(manifest["atom_ids"][name+"_Li"])-1
                    pairs = np.array(list(itertools.combinations(sulfur, 2)))
                    _, ss = find_mic(positions[pairs[:,1]]-positions[pairs[:,0]], cell, True)
                    if name not in original_pairs:
                        original_pairs[name] = ss < 2.6
                        if not original_pairs[name].any():
                            raise RuntimeError("No initially close polysulfide S pairs")
                    _, lis = find_mic((positions[li,None,:]-positions[sulfur][None,:,:]).reshape(-1,3), cell, True)
                    nearest = lis.reshape(len(li),6).min(axis=1)
                    old_distances = [nearest[np.flatnonzero(li == atom)[0]] for atom in original_li]
                    _, central = find_mic(positions[sulfur[3]]-positions[sulfur[2]], cell, True)
                    t = physical_time(step)
                    row = [step, t, name, float(central), float(ss[original_pairs[name]].max()), *map(float, old_distances), ";".join(map(str, li[nearest < 3.5]+1))]
                    writer.writerow(row)
                    measurements.append(row[:8])
        for name in ["LS01", "LS02"]:
            data = [r for r in measurements if r[2] == name]
            for ss_cutoff in [2.6,2.8,3.0]:
                for li_cutoff in [3.5,4.0,4.5]:
                    times = [r[1] for r in data]
                    cleavage = first_persistent(times, [r[4] for r in data], ss_cutoff)
                    departures = [first_persistent(times, [r[col] for r in data], li_cutoff) for col in [5,6]]
                    departures = [t for t in departures if t is not None]
                    departure = min(departures) if departures else None
                    summaries.append(dict(replica=folder.name, polysulfide=name, ss_cutoff_A=ss_cutoff,
                          li_cutoff_A=li_cutoff, duration_required_ps=1.0, frames=len(times),
                          last_observation_ps=times[-1] if times else None,
                          li_departure_ps=departure, ss_separation_ps=cleavage,
                          delay_ps=cleavage-departure if cleavage is not None and departure is not None else None,
                          replica_complete=(folder / "complete.json").exists(),
                          failed_attempts=len(failed_attempts),
                          volume_warning_frames=len(volume_warnings),
                          stability_review_required=bool(failed_attempts or volume_warnings),
                          status="pending" if not times else ("right-censored" if cleavage is None else "geometric-separation-observed")))
    save_csv(WORK / "replicas/replica_event_summary.csv", summaries)


if __name__ == "__main__":
    main()
