#!/usr/bin/env python3
"""Generate four independent Packmol inputs and a fixed original-protocol manifest."""
import json
import shutil
import time
from common import ROOT, WORK, stage_definitions, save_json, sha256


def main():
    source = ROOT / "AIMD/DME_DOL_1MLiTFSI_1MLi2S6/20x"
    for replica in range(1, 5):
        folder = WORK / "replicas" / f"R{replica:02d}"
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir(parents=True)
        parts = ["tolerance 2.0", "filetype pdb", "output packed.pdb", "randominitialpoint", f"seed {190000 + replica}"]
        counts = [("DOL.pdb", 143), ("DME.pdb", 96), ("TFSI-.pdb", 20), ("Li+.pdb", 20), ("Li2S6.pdb", 2)]
        for name, number in counts:
            shutil.copy2(source / name, folder / name)
            parts.extend([f"structure {name}", f"  number {number}", "  inside box 0. 0. 0. 31.4193985 31.4193985 31.4193985", "end structure"])
        (folder / "packmol.inp").write_text("\n".join(parts) + "\n")
        save_json(folder / "manifest.json", dict(replica=replica, natoms=3445,
                  composition=dict(counts), initial_cell_A=33.4193985,
                  packing_seed=190000+replica, velocity_seed=290000+replica,
                  thermostat_seed=390000+replica, stages=stage_definitions(),
                  deadline_unix=time.time()+14*86400,
                  minimization=dict(optimizer="FIRE", fmax_eV_A=2.0, max_steps=500, relax_cell=True),
                  atom_ids=dict(LS01_S=list(range(3430,3436)), LS01_Li=[3436,3437], LS02_S=list(range(3438,3444)), LS02_Li=[3444,3445]),
                  template_sha256={name:sha256(folder/name) for name, _ in counts},
                  output_changes="Dump coordinates, image flags and velocities every 10 integration steps; no every-step DCD. Thermo every 100 steps, restart every 5000.",
                  caveat="Independent replica of nominal successful protocol, not bitwise reproduction of failed/restarted historical branches."))
    save_json(WORK / "replicas/event_definitions.json", dict(
              ss_separation_cutoff_A=2.8, ss_sensitivity_A=[2.6,2.8,3.0],
              li_departure_nearest_S_cutoff_A=4.0, li_sensitivity_A=[3.5,4.0,4.5],
              persistence_ps=1.0, scope="Geometric event definitions, not bond order or radical identification",
              identity="Track original six S atoms and original Li plus all electrolyte Li contacts",
              censoring="No-event trajectories remain right-censored; two PS units share a replica and are not independent boxes"))


if __name__ == "__main__":
    main()
