#!/usr/bin/env python3
"""Export every S-S distance, without equating a distance to a bond order."""
import argparse
import csv
import gzip
import itertools
from collections import Counter, defaultdict
import numpy as np
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
from common import TRAINING, WORK, save_json, save_csv


def sulfur_membership(atoms):
    sulfur = np.flatnonzero(atoms.numbers == 16)
    oxygen = np.flatnonzero(atoms.numbers == 8)
    labels = {}
    for s in sulfur:
        _, distances = find_mic(atoms.positions[oxygen] - atoms.positions[s], atoms.cell, atoms.pbc)
        # TFSI has two covalently bound oxygens; membership is fixed at block start.
        labels[int(s)] = "TFSI" if np.count_nonzero(distances < 1.9) >= 2 else "PS"
    return sulfur, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output = WORK / "training_coverage"
    output.mkdir(parents=True, exist_ok=True)
    source = TRAINING / "train.traj"
    hist = defaultdict(lambda: np.zeros(400, dtype=np.int64))
    edges = np.linspace(0, 20, 401)
    natoms = Counter()
    summaries = []
    with Trajectory(source) as trajectory, gzip.open(output / "all_training_ss_distances.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["training_frame_0based", "source_block", "source_frame_0based", "atom_i_0based", "atom_j_0based", "membership_i", "membership_j", "distance_A", "initial_pair_below_2p6A"])
        total = min(len(trajectory), args.limit or len(trajectory))
        for frame in range(total):
            atoms = trajectory[frame]
            natoms[len(atoms)] += 1
            if frame % 6000 == 0:
                sulfur, membership = sulfur_membership(atoms)
                pairs = np.asarray(list(itertools.combinations(sulfur, 2)), dtype=int)
                _, first = find_mic(atoms.positions[pairs[:, 1]] - atoms.positions[pairs[:, 0]], atoms.cell, atoms.pbc)
                original_close = first < 2.6
                block = frame // 6000
                expected = ("nonLiPS" if block % 2 == 0 else "LiPS") + f"_box{block // 2 + 1}"
                summaries.append(dict(block=block, inferred_archive_name=expected, natoms=len(atoms),
                                      sulfur_indices=sulfur.tolist(), membership=membership,
                                      mapping_basis="Historical assembly order; block identity not a new independent split"))
                print("Starting", expected, "natoms", len(atoms), flush=True)
            _, distances = find_mic(atoms.positions[pairs[:, 1]] - atoms.positions[pairs[:, 0]], atoms.cell, atoms.pbc)
            for pair, distance, initially_close in zip(pairs, distances, original_close):
                i, j = map(int, pair)
                writer.writerow([frame, expected, frame % 6000, i, j, membership[i], membership[j], f"{distance:.8f}", int(initially_close)])
            for category in ["PS-PS", "PS-TFSI", "TFSI-TFSI"]:
                mask = np.array(["-".join(sorted([membership[int(i)], membership[int(j)]])) == category for i, j in pairs])
                hist[category] += np.histogram(distances[mask], edges)[0]
                hist[category + " initial-close"] += np.histogram(distances[mask & original_close], edges)[0]
    rows = [dict(distance_A=float((edges[i] + edges[i + 1]) / 2), **{k: int(v[i]) for k, v in hist.items()}) for i in range(len(edges) - 1)]
    save_csv(output / "distance_histograms.csv", rows)
    save_json(output / "manifest.json", dict(source=str(source), frames=total, complete=total == len(trajectory),
              natoms_counts=dict(natoms), blocks=summaries, units="Angstrom", pbc="ASE minimum image",
              caveat="Original molecular membership is geometric at the first frame of each block; no radical or bond-order assignment."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
    centers = (edges[:-1] + edges[1:]) / 2
    for category, color in [("PS-PS", "#2678b2"), ("TFSI-TFSI", "#dc8431"), ("PS-TFSI", "#478b51")]:
        for ax, key in zip(axes, [category, category + " initial-close"]):
            counts = hist[key]
            if counts.sum():
                ax.plot(centers, counts / counts.sum() / (edges[1] - edges[0]), label=category, color=color)
    for ax in axes:
        ax.set(xlabel="S-S distance (Angstrom)", ylabel="Probability density")
        ax.legend(framealpha=1)
    axes[0].set_title("All sulfur pairs")
    axes[1].set(title="Pairs initially within 2.6 Angstrom", xlim=(1.5, 5))
    fig.savefig(output / "training_ss_coverage.png", dpi=400)
    fig.savefig(output / "training_ss_coverage.pdf")


if __name__ == "__main__":
    main()
