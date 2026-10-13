#!/usr/bin/env python3
"""Reconstruct the historical inputs; validation overlaps the Testing archive."""
from pathlib import Path
import argparse
import json
from ase.io.trajectory import Trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Directory containing extracted Training/ and Testing/.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists.")
    args.output.mkdir(parents=True)
    frames = 0
    with Trajectory(args.output / "historical_train.traj", "w") as out:
        for box in range(1, 11):
            for chemistry in ("nonLiPS", "LiPS"):
                with Trajectory(args.data_root / "Training" / f"{chemistry}_box{box}.traj") as source:
                    for atoms in source:
                        out.write(atoms)
                        frames += 1
    matches = json.loads((Path(__file__).parent / "historical_validation_matches.json").read_text())
    with Trajectory(args.output / "historical_validation.traj", "w") as out:
        for row in matches:
            match = row["source_matches"][0]
            with Trajectory(args.data_root / match["file"]) as source:
                out.write(source[match["frame"]])
    print(f"Reconstructed {frames} training and {len(matches)} historical validation frames.")
    print("WARNING: these validation frames come from Testing; do not count them as independent tests.")


if __name__ == "__main__":
    main()
