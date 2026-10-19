#!/usr/bin/env python3
"""Convergence-gated density time series and descriptive block sensitivity."""
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
from ase.io import read
from common import WORK, ELEMENTS, DFT_MASSES, save_csv, save_json


def vasp_volumes(path):
    values = []
    for _, element in ET.iterparse(path, events=["end"]):
        if element.tag == "calculation":
            basis = element.find("structure/crystal/varray[@name='basis']")
            if basis is None:
                raise ValueError("Missing ionic-step cell")
            values.append(abs(np.linalg.det([[float(x) for x in v.text.split()] for v in basis])))
            element.clear()
    return np.arange(1, len(values)+1)*0.001, np.asarray(values)


def lammps_volumes(path):
    rows = []
    fields = None
    for line in path.read_text().splitlines():
        words = line.split()
        if words[:2] == ["Step", "Time"]:
            fields = words
        elif fields and len(words) == len(fields):
            try:
                values = list(map(float, words))
                rows.append((values[fields.index("Time")], values[fields.index("Volume")]))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("No ML thermo volumes")
    data = np.array(rows)
    return data[:,0], data[:,1]


def block_statistics(times, values, block_ps):
    labels = np.floor((times-times[0]+1e-9)/block_ps).astype(int)
    full = int((times[-1]-times[0])/block_ps)
    means = np.array([np.mean(values[labels == k]) for k in range(full) if np.any(labels == k)])
    return dict(block_ps=block_ps, nblocks=len(means),
                block_sem=float(means.std(ddof=1)/np.sqrt(len(means))) if len(means) >= 3 else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discard-ps", type=float, default=0.5)
    args = parser.parse_args()
    series, summary = [], []
    for chemistry in ["PS", "nonPS"]:
        atoms = read(WORK / "density" / chemistry / "retry02/POSCAR")
        natural_mass = atoms.get_masses().sum()
        masses = dict(zip(ELEMENTS, DFT_MASSES))
        simulation_mass = sum(masses[s] for s in atoms.get_chemical_symbols())
        for method, relative in [("DFT", "npt_2ps"), ("MLIP", "ml_pilot")]:
            folder = WORK / "density" / chemistry / relative
            if method == "DFT":
                status_file = folder / "pilot_status.json"
                if not status_file.exists():
                    continue
                status = json.loads(status_file.read_text())
                if status["returncode"] or not status["final_timing_present"] or status["ediff_reached_count"] != 2000:
                    continue
                times, volumes = vasp_volumes(folder / "vasprun.xml")
                if len(times) != 2000:
                    raise RuntimeError("DFT ionic-frame count mismatch")
            else:
                if not (folder / "final.restart").exists():
                    continue
                times, volumes = lammps_volumes(folder / "log.lammps")
            for convention, mass in [("natural_element_masses", natural_mass), ("AIMD_simulation_masses", simulation_mass)]:
                density = mass * 1.66053906660 / volumes
                series.extend(dict(chemistry=chemistry, method=method, mass_convention=convention,
                                   time_ps=float(t), volume_A3=float(v), density_g_cm3=float(d))
                              for t,v,d in zip(times,volumes,density))
                mask = times >= args.discard_ps
                if mask.sum() < 10:
                    continue
                for block in [0.2,0.5]:
                    stats = block_statistics(times[mask], density[mask], block)
                    summary.append(dict(chemistry=chemistry, method=method, mass_convention=convention,
                        mean_density_g_cm3=float(density[mask].mean()), discard_ps=args.discard_ps,
                        last_time_ps=float(times[-1]), **stats,
                        interpretation="Descriptive short-run statistic only; stationarity and independent blocks not established"))
    if not series:
        save_json(WORK / "density/comparison_status.json", dict(status="pending: no completed converged matched runs"))
        print("No converged completed comparison yet; no density claims generated")
        return
    save_csv(WORK / "density/density_timeseries.csv", series)
    save_csv(WORK / "density/density_comparison_provisional.csv", summary)
    save_json(WORK / "density/comparison_status.json", dict(status="provisional short-run comparison", warning="Natural elemental masses may differ from experimental isotope composition. No equilibrium-density or causal diffusion claim."))


if __name__ == "__main__":
    main()
