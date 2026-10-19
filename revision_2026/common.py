"""Shared paths and small, provenance-preserving utilities for the revision."""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path(os.environ.get("MLIP_PROJECT_ROOT", "/sdf/group/mli/MLIP_Li_S_Fl_Elec"))
TRAINING = ROOT / "AIMD/chgnet/m3gnet_with_stress"
BASE = TRAINING / "lammps-m3gnet/smaller_model/20x"
REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("REVISION_WORK", str(BASE / "JPCRevision/reviewer_revision_20261019")))
MODEL = REPO / "models/fine_tuned_m3gnet_model"
PATH = BASE / "PS/dissolution_mechanism/dft_fixed_environment_path"
SPIN = PATH / "ml_relaxation_dft_recheck/full_reaction_path/spin_polarized_frame_F"
ELEMENTS = ["C", "F", "H", "Li", "N", "O", "S"]
DFT_MASSES = [12.011, 18.998, 2.000, 7.010, 14.001, 16.000, 32.066]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Refusing an empty result table: " + str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stage_definitions():
    # Physical times and masses come from executable inputs, not their comments.
    return [
        dict(name="01_initial", start=0, end=100000, dt=0.002, t0=50, t1=300, ensemble="langevin_nph", hmass=3.016049),
        dict(name="02_heat", start=100000, end=300000, dt=0.001, t0=300, t1=350, ensemble="langevin_nph", hmass=3.016049),
        dict(name="03_hold", start=300000, end=400000, dt=0.001, t0=350, t1=350, ensemble="langevin_nph", hmass=3.016049),
        dict(name="04_cool", start=400000, end=600000, dt=0.001, t0=350, t1=300, ensemble="langevin_nph", hmass=3.016049),
        dict(name="05_npt", start=600000, end=800000, dt=0.0005, t0=300, t1=300, ensemble="npt", hmass=1.00783),
        dict(name="06_nvt", start=800000, end=1400000, dt=0.0005, t0=300, t1=300, ensemble="nvt", hmass=1.00783),
    ]


def physical_time(step):
    elapsed = 0.0
    for stage in stage_definitions():
        elapsed += max(0, min(step, stage["end"]) - stage["start"]) * stage["dt"]
    return elapsed
