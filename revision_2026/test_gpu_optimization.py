#!/usr/bin/env python3
"""Require numerical parity before opting any replica into the faster driver."""
import cProfile
import importlib.util
import os
from pathlib import Path
import pstats
import subprocess
import time
import numpy as np
from ase.io import read
from common import ROOT, REPO, WORK, MODEL, PATH, save_json
from fast_graph import install


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = WORK / "gpu_optimization" / os.environ["SLURM_JOB_ID"]
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("archive", ROOT / "advancesoft-lammps/src/ML-GNNP/gnnp_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.gnnp_initialize("matgl", str(MODEL))
    paths = [REPO / f"reproducibility/reaction/snapshots/ps_{t}ps.extxyz" for t in [100,260,330,340]]
    paths += [PATH / "images/20_step300000/POSCAR"]
    states = [read(p) for p in paths]
    references = []
    timings = []
    profile = cProfile.Profile()

    def evaluate(atoms):
        return driver.gnnp_get_energy_forces_stress(atoms.cell.array.tolist(), atoms.numbers.tolist(), atoms.positions.tolist(), True)

    for atoms in states:
        references.append(evaluate(atoms))
    profiled = states[2].copy()
    profiled.positions[0,0] += 1e-5
    profile.runcall(evaluate, profiled)
    for variant in ["original", "vectorized_triplets", "vectorized_and_first_derivatives"]:
        if variant != "original":
            install(inference_grad=variant.endswith("derivatives"))
        discrepancies = []
        for path, atoms, reference in zip(paths, states, references):
            driver.myAtoms = None
            result = evaluate(atoms)
            de = abs(float(result[0])-float(reference[0]))/len(atoms)
            df = float(np.max(np.abs(np.asarray(result[1])-np.asarray(reference[1]))))
            ds = float(np.max(np.abs(np.asarray(result[2])-np.asarray(reference[2]))))
            discrepancies.append(dict(path=str(path), energy_difference_eV_atom=de,
                                      max_force_difference_eV_A=df, max_stress_difference_eV_A3=ds))
            if de > 1e-4 or df > 1e-4 or ds > 1e-7:
                save_json(output / "parity_failure.json", dict(variant=variant, discrepancies=discrepancies))
                raise RuntimeError("Optimized evaluator failed numerical parity")
        times = []
        atoms = states[2].copy()
        for iteration in range(23):
            atoms.positions[0,0] += 1e-5
            torch.cuda.synchronize()
            start = time.perf_counter()
            evaluate(atoms)
            torch.cuda.synchronize()
            if iteration >= 3:
                times.append(time.perf_counter()-start)
        timings.append(dict(variant=variant, median_seconds=float(np.median(times)), discrepancies=discrepancies))
        print(timings[-1], flush=True)
    profile.dump_stats(str(output / "original.cprofile"))
    with (output / "original_cprofile.txt").open("w") as handle:
        pstats.Stats(profile, stream=handle).sort_stats("cumulative").print_stats(45)
    # The legacy calculator remains loaded only in this process. Child LAMMPS
    # starts fresh for both drivers, including input parsing and embedded Python.
    smoke_results = []
    for name, driver_path in [("original", ROOT / "advancesoft-lammps/src/ML-GNNP"),
                              ("optimized", Path(__file__).parent / "optimized_driver")]:
        folder = output / name
        folder.mkdir()
        inp = folder / "input.lammps"
        inp.write_text(f'''units metal
atom_style atomic
boundary p p p
read_data {REPO}/reproducibility/reaction/snapshots/ps_330ps.data
pair_style gnnp/gpu {driver_path}
pair_coeff * * matgl {MODEL} C F H Li N O S
timestep 0.0005
fix 1 all nve
thermo 1
thermo_style custom step pe temp press vol
run 20
write_data final.data
''')
        with (folder / "screen.log").open("w") as handle:
            subprocess.run([str(ROOT / "advancesoft-lammps/build/lmp"), "-in", str(inp)], cwd=folder, stdout=handle, stderr=subprocess.STDOUT, check=True)
        smoke_results.append(read(folder / "final.data", format="lammps-data", atom_style="atomic"))
    difference = float(np.max(np.abs(smoke_results[0].positions-smoke_results[1].positions)))
    if difference > 1e-3:
        raise RuntimeError(f"Short dynamics diverged beyond smoke tolerance: {difference}")
    speedup = timings[0]["median_seconds"] / timings[-1]["median_seconds"]
    save_json(output / "results.json", dict(timings=timings, speedup=speedup,
              short_trajectory_max_position_difference_A=difference, gpu=torch.cuda.get_device_name(0)))
    if speedup > 1.1:
        save_json(WORK / "gpu_optimization/selected_driver.json", dict(
                  driver=str(Path(__file__).resolve().parent / "optimized_driver"),
                  tests=str(output / "results.json"), speedup=speedup,
                  model_unchanged=True, experimental_inference_adapter=True))


if __name__ == "__main__":
    main()
