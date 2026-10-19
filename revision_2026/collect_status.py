#!/usr/bin/env python3
"""Collect live job/pilot evidence; never infer convergence from a job name."""
from datetime import datetime, timezone
import json
import os
import subprocess
from common import WORK, save_json


def main():
    status = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), pilots=[], replicas=[])
    try:
        query = subprocess.run(["squeue", "-u", os.environ.get("USER", "ssv5"), "-h", "-o", "%i|%j|%T|%M|%R"],
                               text=True, capture_output=True, timeout=20)
        status["squeue"] = query.stdout.splitlines()
        status["squeue_error"] = query.stderr if query.returncode else None
    except (OSError, subprocess.TimeoutExpired) as error:
        status["squeue_error"] = str(error)
    for category in ["hse", "density"]:
        for path in sorted((WORK / category).glob("*/*/pilot_status.json")):
            item = json.loads(path.read_text())
            item["folder"] = str(path.parent.relative_to(WORK))
            status["pilots"].append(item)
    for folder in sorted((WORK / "replicas").glob("R0*")):
        steps = [int(p.name.split(".")[1]) for p in folder.glob("*/checkpoint.*.restart")]
        status["replicas"].append(dict(replica=folder.name, latest_checkpoint=max(steps, default=0),
            complete=(folder / "complete.json").exists(),
            failed_attempts=[str(p.relative_to(folder)) for p in folder.glob("*/attempt_*/failed.json")]))
    status["interpretation"] = "Job completion is not scientific validation. Pilots are not equilibrium densities; static snapshot energies are not free energies. Filippo's FES and within-state analysis remain external work."
    save_json(WORK / "status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
