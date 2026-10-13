#!/usr/bin/env python3
"""Stitch the successful PS pre-equilibration restart lineage.

Failed branches are retained on disk but are excluded here after the checkpoint
from which each successful replacement run was launched. Duplicate frames at
restart boundaries are also removed.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PS_DIR = ROOT.parent
OUT = ROOT / "stitched_valid_preequilibration"
TRAJECTORY = OUT / "stitched_valid_preequilibration_000000_600000.lammpstrj"
MANIFEST = OUT / "stitched_valid_preequilibration_manifest.csv"

EXPECTED_ATOMS = 3445
EXPECTED_STEP_INTERVAL = 10
MAX_VALID_CELL_A = 40.0


@dataclass(frozen=True)
class Segment:
    label: str
    path: Path
    first_step: int
    last_step: int


SEGMENTS = [
    Segment(
        "initial_npt_50_to_300K",
        PS_DIR / "xyz.lammpstrj",
        0,
        100000,
    ),
    Segment(
        "heat_original_before_restart_200000",
        PS_DIR / "HU_300-350K_1fs" / "prev_job" / "heat_300to400.lammpstrj",
        100000,
        200000,
    ),
    Segment(
        "heat_restarted_from_200000",
        PS_DIR / "HU_300-350K_1fs" / "heat_300to400.lammpstrj",
        200000,
        300000,
    ),
    Segment(
        "hold_original_before_restart",
        PS_DIR / "Main_350-350K_1fs" / "prev" / "main_350.lammpstrj",
        300000,
        320000,
    ),
    Segment(
        "hold_restarted_from_320000",
        PS_DIR / "Main_350-350K_1fs" / "prev_2" / "main_350.lammpstrj",
        320000,
        390000,
    ),
    Segment(
        "hold_restarted_from_390000",
        PS_DIR / "Main_350-350K_1fs" / "main_350.lammpstrj",
        390000,
        400000,
    ),
    Segment(
        "cool_original_before_restart",
        PS_DIR / "CD_350-300K_1fs" / "prev" / "heat_350to300.lammpstrj",
        400000,
        450000,
    ),
    Segment(
        "cool_restarted_from_450000",
        PS_DIR / "CD_350-300K_1fs" / "heat_350to300.lammpstrj",
        450000,
        600000,
    ),
]


def iter_raw_frames(path: Path):
    with path.open() as handle:
        while True:
            first = handle.readline()
            if not first:
                return
            if not first.startswith("ITEM: TIMESTEP"):
                continue
            step_line = handle.readline()
            step = int(step_line)
            number_header = handle.readline()
            atom_count_line = handle.readline()
            atom_count = int(atom_count_line)
            box_header = handle.readline()
            bounds = [handle.readline() for _ in range(3)]
            atom_header = handle.readline()
            atoms = [handle.readline() for _ in range(atom_count)]
            if any(line == "" for line in atoms):
                raise RuntimeError(f"Truncated frame at step {step} in {path}")
            raw = [
                first,
                step_line,
                number_header,
                atom_count_line,
                box_header,
                *bounds,
                atom_header,
                *atoms,
            ]
            lengths = []
            for line in bounds:
                lo, hi = map(float, line.split()[:2])
                lengths.append(hi - lo)
            yield step, atom_count, lengths, raw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    partial = TRAJECTORY.with_suffix(TRAJECTORY.suffix + ".partial")
    rows = []
    last_written = None
    total_frames = 0

    with partial.open("w") as output:
        for segment in SEGMENTS:
            if not segment.path.exists():
                raise FileNotFoundError(segment.path)
            selected_steps = []
            box_lengths = []
            for step, atom_count, lengths, raw in iter_raw_frames(segment.path):
                if step < segment.first_step:
                    continue
                if step > segment.last_step:
                    break
                if atom_count != EXPECTED_ATOMS:
                    raise RuntimeError(
                        f"{segment.label}: expected {EXPECTED_ATOMS} atoms, found {atom_count} at step {step}"
                    )
                if max(lengths) >= MAX_VALID_CELL_A:
                    raise RuntimeError(
                        f"{segment.label}: rejected expanded cell {lengths} at step {step}"
                    )
                if last_written is not None and step == last_written:
                    continue
                if last_written is not None and step != last_written + EXPECTED_STEP_INTERVAL:
                    raise RuntimeError(
                        f"Non-contiguous stitch: step {last_written} followed by {step} in {segment.label}"
                    )
                output.writelines(raw)
                last_written = step
                selected_steps.append(step)
                box_lengths.extend(lengths)
                total_frames += 1
            if not selected_steps:
                raise RuntimeError(f"No new frames selected from {segment.path}")
            rows.append(
                {
                    "segment": segment.label,
                    "source": str(segment.path),
                    "first_written_step": selected_steps[0],
                    "last_written_step": selected_steps[-1],
                    "frames_written": len(selected_steps),
                    "min_cell_A": min(box_lengths),
                    "max_cell_A": max(box_lengths),
                }
            )

    if last_written != 600000:
        raise RuntimeError(f"Stitch ended at {last_written}, expected 600000")
    os.replace(partial, TRAJECTORY)
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"trajectory: {TRAJECTORY}")
    print(f"manifest: {MANIFEST}")
    print(f"frames: {total_frames}")
    print(f"steps: 0 to {last_written} every {EXPECTED_STEP_INTERVAL}")


if __name__ == "__main__":
    main()
