#!/usr/bin/env python3
"""Document the usable new-composition benchmark without relabeling old tests."""
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from ase.io import read
from common import BASE, REPO, WORK, save_json, save_csv, sha256


def main():
    source = BASE / "JPCRevision/transferability"
    metric_file = source / "analysis/publication_no_exploded_with_T05/dft_energies_publication_no_exploded.csv"
    metrics = list(csv.DictReader(metric_file.open()))
    mapping = {"T01_nonPS_baseline": "T01", "T02_PS_baseline": "T02",
               "T04_DME_rich_PS": "T03", "T05_Li2S4_speciation": "T04"}
    training_counts = set(map(int, json.loads((WORK / "training_coverage/manifest.json").read_text())["natoms_counts"]))
    records = []
    hashes = []
    for row in metrics:
        system = row["system_id"]
        label = mapping[system]
        samples = "sampled_poscars_T05" if system.startswith("T05") else "sampled_poscars_clean_final_no_exploded"
        path = source / samples / system / f"POSCAR_{int(row['frame_id']):04d}"
        atoms = read(path, format="vasp")
        if len(atoms) != int(row["natoms"]):
            raise RuntimeError("Benchmark/POSCAR atom-count mismatch")
        digest = hashlib.sha256(atoms.numbers.astype('<i8').tobytes() +
                 np.round(atoms.cell.array, 8).astype('<f8').tobytes() +
                 np.round(atoms.positions, 8).astype('<f8').tobytes()).hexdigest()
        hashes.append(digest)
        records.append(dict(published_label=label, source_system=system, frame_id=row["frame_id"],
            natoms=len(atoms), source=str(path), poscar_sha256=sha256(path), geometry_hash=digest,
            excluded_from_final_stage_by_atom_count=len(atoms) not in training_counts))
    overlap = json.loads((REPO / "training/historical_validation_matches.json").read_text())
    save_csv(WORK / "training_audit/transferability_provenance.csv", records)
    save_json(WORK / "training_audit/split_status.json", dict(
        historical_validation_frames=len(overlap), historical_matching_trajectories=dict(Counter(
            filename for row in overlap for filename in {hit["file"] for hit in row["source_matches"]})),
        transferability_counts=dict(Counter(r["published_label"] for r in records)),
        transferability_unique_ordered_geometries=len(set(hashes)),
        all_excluded_from_final_stage_by_atom_count=all(r["excluded_from_final_stage_by_atom_count"] for r in records),
        mapping=mapping, benchmark_csv_sha256=sha256(metric_file),
        limitation="The old Testing trajectories are not independent of training-time validation. New transferability cells cannot be identical to final-stage 146/154-atom geometries; earlier adaptation caches are not fully reconstructed. This audit does not repair energy normalization or validate the new-system metrics."))


if __name__ == "__main__":
    main()
