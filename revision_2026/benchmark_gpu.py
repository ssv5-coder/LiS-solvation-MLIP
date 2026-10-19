#!/usr/bin/env python3
"""Benchmark the archived driver without changing its model or implementation."""
import argparse
import importlib.util
import json
import os
import subprocess
import time
import numpy as np
from ase.io import read
from common import ROOT, REPO, MODEL, WORK, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    import torch
    import matgl
    import dgl
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; refusing CPU fallback")
    folder = WORK / "gpu_benchmark" / str(os.environ.get("SLURM_JOB_ID", "local"))
    folder.mkdir(parents=True, exist_ok=True)
    driver_path = ROOT / "advancesoft-lammps/src/ML-GNNP/gnnp_driver.py"
    spec = importlib.util.spec_from_file_location("gnnp_driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    print("Driver entry points:", [n for n in dir(driver) if n.startswith("gnnp_")], flush=True)
    driver.gnnp_initialize("matgl", str(MODEL))
    records = []
    for threads in [1, 4]:
        torch.set_num_threads(threads)
        for snapshot in ["100", "340"]:
            atoms = read(REPO / f"reproducibility/reaction/snapshots/ps_{snapshot}ps.extxyz")
            cell = atoms.cell.array.tolist()
            numbers = atoms.numbers.tolist()
            positions = atoms.positions.copy()
            timings = []
            for iteration in range(args.repeats + 3):
                # Bypass ASE's identical-state cache without changing the test's scale.
                moved = positions.copy()
                moved[0, 0] += 1e-5 * iteration
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = driver.gnnp_get_energy_forces_stress(cell, numbers, moved.tolist(), True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if not all(np.isfinite(np.asarray(x)).all() for x in result):
                    raise RuntimeError("Nonfinite energy, force, or stress")
                if iteration >= 3:
                    timings.append(elapsed)
            row = dict(snapshot_ps=int(snapshot), threads=threads,
                       median_seconds=float(np.median(timings)), mean_seconds=float(np.mean(timings)),
                       predicted_1400000_step_days=float(np.mean(timings) * 1400000 / 86400),
                       final_energy_eV=float(result[0]))
            records.append(row)
            print(json.dumps(row), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for iteration in range(5):
            moved = positions.copy()
            moved[0, 0] += 1e-4 * (iteration + 1)
            driver.gnnp_get_energy_forces_stress(cell, numbers, moved.tolist(), True)
            prof.step()
    (folder / "profile.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    prof.export_chrome_trace(str(folder / "profile.json"))
    save_json(folder / "benchmark.json", dict(records=records, gpu=torch.cuda.get_device_name(0),
              torch=torch.__version__, matgl=matgl.__version__, dgl=dgl.__version__,
              cuda=torch.version.cuda, model=str(MODEL), driver=str(driver_path),
              scope="Archived Python inference driver; excludes LAMMPS integration and I/O"))
    from audit_energy_convention import audit
    audit(driver)
    # Exercise the actual executable, including its Python embedding and stress units.
    smoke = folder / "lammps_smoke.in"
    smoke.write_text(f'''units metal
atom_style atomic
boundary p p p
read_data {REPO}/reproducibility/reaction/snapshots/ps_330ps.data
pair_style gnnp/gpu {ROOT}/advancesoft-lammps/src/ML-GNNP
pair_coeff * * matgl {MODEL} C F H Li N O S
timestep 0.0005
fix 1 all nve
thermo 1
thermo_style custom step pe temp press vol
run 10
''')
    with (folder / "lammps_smoke.out").open("w") as handle:
        subprocess.run([str(ROOT / "advancesoft-lammps/build/lmp"), "-in", str(smoke)], cwd=folder, stdout=handle, stderr=subprocess.STDOUT, check=True)
    save_json(folder / "passed.json", dict(driver_test=True, lammps_steps=10,
              warning="Execution smoke test only, not DFT accuracy or long-time stability validation"))


if __name__ == "__main__":
    main()
