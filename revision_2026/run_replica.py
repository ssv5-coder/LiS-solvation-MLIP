#!/usr/bin/env python3
"""Run/resume nominal original stages, preserving every restart branch."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import ROOT, MODEL, WORK, save_json, sha256


def make_input(stage, initial, seed, velocity_seed, attempt, driver=None):
    driver = driver or ROOT / "advancesoft-lammps/src/ML-GNNP"
    read_command = f'read_data "{initial}"' if stage["start"] == 0 and initial.name == "initial.data" else f'read_restart "{initial}"'
    initialization = f"velocity all create 50.0 {velocity_seed} dist gaussian\n" if read_command.startswith("read_data") else ""
    ensemble = stage["ensemble"]
    if ensemble == "langevin_nph":
        fixes = f'fix 1 all langevin {stage["t0"]} {stage["t1"]} 1.0 {seed}\nfix 2 all nph iso 1.01325 1.01325 5.0\nfix_modify 2 temp myTemp press pTot'
    elif ensemble == "npt":
        fixes = 'fix 1 all npt temp 300.0 300.0 0.2 iso 1.01325 1.01325 2.0\nfix_modify 1 press pTot temp myTemp'
    else:
        fixes = 'fix 1 all nvt temp 300.0 300.0 0.1'
    # run start/stop preserves the ramp across checkpoint continuations.
    return f'''units metal
boundary p p p
atom_style atomic
{read_command}
mass 3 {stage["hmass"]}
pair_style gnnp/gpu {driver}
pair_coeff * * matgl {MODEL} C F H Li N O S
compute myTemp all temp
compute pTot all pressure myTemp
thermo 100
thermo_style custom step time cpu pe ke etotal temp press vol density
thermo_modify flush yes lost error
timestep {stage["dt"]}
{initialization}{fixes}
dump coords all custom 10 {attempt}/trajectory.lammpstrj id type x y z ix iy iz vx vy vz
dump_modify coords sort id format float %.10g
restart 5000 checkpoint.*.restart
run {stage["end"]} upto start {stage["start"]} stop {stage["end"]}
write_restart final.restart
write_data final.data
print "REVISION_STAGE_COMPLETE $(step:%.0f)"
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=int, required=True, choices=range(1,5))
    parser.add_argument("--wall-seconds", type=int, default=84600)
    args = parser.parse_args()
    folder = WORK / "replicas" / f"R{args.replica:02d}"
    manifest = json.loads((folder / "manifest.json").read_text())
    evaluator_file = folder / "evaluator.json"
    if evaluator_file.exists():
        evaluator = json.loads(evaluator_file.read_text())
        for path, expected in evaluator["checksums"].items():
            if sha256(path) != expected:
                raise RuntimeError("Evaluator changed during a replica: " + path)
    else:
        selection = WORK / "gpu_optimization/selected_driver.json"
        evaluator = json.loads(selection.read_text()) if selection.exists() else dict(
            driver=str(ROOT / "advancesoft-lammps/src/ML-GNNP"), model_unchanged=True)
        files = [Path(evaluator["driver"]) / "gnnp_driver.py", *MODEL.glob("*")]
        if selection.exists():
            files.append(Path(__file__).with_name("fast_graph.py"))
        evaluator["checksums"] = {str(p): sha256(p) for p in files if p.is_file()}
        save_json(evaluator_file, evaluator)
    if time.time() > manifest["deadline_unix"]:
        raise RuntimeError("Two-week deadline reached; no automatic extension")
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *args: stop.__setitem__(0, True))
    signal.signal(signal.SIGUSR1, lambda *args: stop.__setitem__(0, True))
    started = time.monotonic()
    if not (folder / "initial.data").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("relax_replica.py")), str(folder)], check=True, cwd=folder)
    previous = folder / "initial.data"
    for index, stage in enumerate(manifest["stages"]):
        directory = folder / stage["name"]
        directory.mkdir(exist_ok=True)
        if (directory / "complete.json").exists():
            previous = directory / "final.restart"
            continue
        checkpoints = sorted(directory.glob("checkpoint.*.restart"), key=lambda p: int(p.name.split(".")[1]))
        checkpoints = [p for p in checkpoints if p.stat().st_size > 10000]
        initial = checkpoints[-1] if checkpoints else previous
        attempt_index = len(list(directory.glob("attempt_*")))
        attempt = directory / f"attempt_{attempt_index:03d}"
        attempt.mkdir()
        seed = manifest["thermostat_seed"] + index*1000 + attempt_index
        inp = attempt / "input.lammps"
        inp.write_text(make_input(stage, initial, seed, manifest["velocity_seed"], attempt, evaluator["driver"]))
        save_json(attempt / "lineage.json", dict(source_restart=str(initial), stage=stage,
                  job_id=os.environ.get("SLURM_JOB_ID"), thermostat_seed=seed, evaluator=evaluator,
                  note="Langevin RNG is re-seeded on restart; future coordinates are not bitwise identical"))
        with (attempt / "screen.log").open("w") as log:
            process = subprocess.Popen([str(ROOT / "advancesoft-lammps/build/lmp"), "-in", str(inp), "-log", str(attempt / "log.lammps")], cwd=directory, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if stop[0] or time.monotonic()-started > args.wall_seconds or time.time() > manifest["deadline_unix"]:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    save_json(attempt / "interrupted.json", dict(reason="time limit or signal", resume="Last complete periodic checkpoint; keep later trajectory tail but exclude it from stitched analysis"))
                    return 75
                time.sleep(10)
        if process.returncode != 0:
            save_json(attempt / "failed.json", dict(returncode=process.returncode, action="Inspect, do not silently restart an unstable branch"))
            raise RuntimeError(f"LAMMPS failed: {attempt}")
        if f'REVISION_STAGE_COMPLETE {stage["end"]}' not in (attempt / "screen.log").read_text(errors="replace"):
            raise RuntimeError("Stage stopped without reaching required final step")
        save_json(directory / "complete.json", dict(step=stage["end"], attempt=str(attempt), restart=str(directory / "final.restart")))
        previous = directory / "final.restart"
    save_json(folder / "complete.json", dict(total_time_ps=1100, final_step=1400000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
