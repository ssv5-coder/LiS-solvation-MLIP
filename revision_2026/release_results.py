#!/usr/bin/env python3
"""Copy only explicit compact revision artifacts; exclude licensed/large outputs."""
import csv
import json
from pathlib import Path
import shutil
from common import WORK, REPO, save_json, sha256


def main():
    destination = REPO / "revision_2026/results"
    patterns = ["status.json", "training_coverage/*.csv.gz",
                "training_coverage/*.csv", "training_coverage/*.json", "training_coverage/*.png",
                "training_coverage/*.pdf", "training_audit/*.csv", "training_audit/*.json",
                "reactive_validation/*.csv", "reactive_validation/*.json", "reactive_validation/*.png",
                "reactive_validation/*.pdf", "density/*.csv", "density/*.json",
                "density/*/*/pilot_status.json", "hse/*/*/pilot_status.json", "spin/*.json", "spin/*.csv",
                "gpu_benchmark/*/benchmark.json", "gpu_benchmark/*/passed.json",
                "gpu_optimization/*/results.json", "gpu_optimization/selected_driver.json",
                "replicas/replica_event_summary.csv", "replicas/R0*/manifest.json",
                "replicas/R0*/packmol.inp", "replicas/R0*/packed.pdb",
                "replicas/R0*/preparation.json", "replicas/R0*/evaluator.json"]
    records = []
    for pattern in patterns:
        for source in sorted(WORK.glob(pattern)):
            if source.stat().st_size > 40_000_000:
                raise ValueError("Unexpected large release artifact: " + str(source))
            target = destination / source.relative_to(WORK)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".csv":
                with source.open(newline="") as input_file, target.open("w", newline="") as output_file:
                    csv.writer(output_file, lineterminator="\n").writerows(csv.reader(input_file))
            else:
                shutil.copy2(source, target)
            records.append(dict(path=str(target.relative_to(destination)), sha256=sha256(target),
                                bytes=target.stat().st_size))
    save_json(destination / "checksums.json", records)
    print(f"Prepared {len(records)} compact release artifacts; no POTCAR, WAVECAR or trajectories")


if __name__ == "__main__":
    main()
