#!/usr/bin/env python3
"""Portable version of the archived final fine-tuning stage.

Input energies must be TOTAL eV, forces eV/A and ASE stresses eV/A^3.
See README.md for historical preprocessing and validation-split limitations.
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True, type=Path)
    parser.add_argument("--validation", nargs="+", required=True, type=Path)
    parser.add_argument("--initial-model", type=Path, default=Path(__file__).resolve().parents[1] / "models/final_stage_initialization")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--energy-label-convention", choices=["total", "historical-per-atom"],
                        default="total", help="Use historical-per-atom only to investigate the archived notebook convention.")
    parser.add_argument("--smoke-test", action="store_true", help="One train and one validation batch; no final model export.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory to prevent stale graph caches.")
    if set(p.resolve() for p in args.train) & set(p.resolve() for p in args.validation):
        parser.error("Training and validation file lists overlap.")

    import numpy as np
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from ase.io import iread
    from ase.units import GPa
    from dgl.data.utils import split_dataset
    from pymatgen.io.ase import AseAtomsAdaptor
    from matgl.apps.pes import Potential
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
    from matgl.utils.training import PotentialLightningModule

    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    potential = Potential.load(args.initial_model)
    model = potential.model
    # Explicit full fine-tuning of all M3GNet network parameters.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    args.output.mkdir(parents=True)
    converter = Structure2Graph(element_types=model.element_types, cutoff=5.0)
    manifest = {}

    def dataset(paths, name):
        structures = []
        labels = {"energies": [], "forces": [], "stresses": []}
        manifest[name] = []
        for path in paths:
            count = 0
            for atoms in iread(str(path), index=":"):
                structures.append(AseAtomsAdaptor.get_structure(atoms))
                # MatGL itself divides both target/predicted total energies by N.
                energy = float(atoms.get_potential_energy())
                if args.energy_label_convention == "historical-per-atom":
                    energy /= len(atoms)
                labels["energies"].append(energy)
                labels["forces"].append(atoms.get_forces().tolist())
                labels["stresses"].append((atoms.get_stress(voigt=False) / GPa).tolist())
                count += 1
            manifest[name].append({"file": str(path), "frames": count})
        if not structures:
            raise ValueError(f"No structures in {name}")
        cache = args.output / (name + "_graphs")
        return MGLDataset(structures=structures, converter=converter, labels=labels,
                          threebody_cutoff=4.0, include_line_graph=True,
                          save_dir=str(cache), raw_dir=str(cache))

    train_set = dataset(args.train, "training")
    val_set = dataset(args.validation, "validation")
    train_set, _, _ = split_dataset(train_set, frac_list=[1, 0, 0], shuffle=True, random_state=42)
    _, val_set, _ = split_dataset(val_set, frac_list=[0, 1, 0], shuffle=True, random_state=42)
    train_loader, val_loader = MGLDataLoader(
        train_data=train_set, val_data=val_set,
        collate_fn=partial(collate_fn_pes, include_line_graph=True, include_stress=True),
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=args.accelerator != "cpu", generator=torch.Generator(device="cpu"))
    module = PotentialLightningModule(
        model=model, element_refs=potential.element_refs.property_offset,
        lr=1e-4, include_line_graph=True, energy_weight=1.0, force_weight=1.0,
        stress_weight=0.01, decay_steps=50, decay_alpha=0.1)

    class StressWeightScheduler(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            pl_module.stress_weight = 0.01 + 0.09 * min(trainer.current_epoch / 19, 1)
            pl_module.log("stress_weight", pl_module.stress_weight, on_epoch=True)

    manifest.update({
        "initial_model": str(args.initial_model), "full_finetuning": True,
        "trainable_network_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "frozen_network_parameters": sum(x.numel() for x in model.parameters() if not x.requires_grad),
        "energy_labels": args.energy_label_convention, "stress_labels": "GPa",
        "epochs": args.epochs, "smoke_test": args.smoke_test,
        "note": "Portable reconstruction, not a claim of bitwise historical reproduction."})
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.epochs,
        accumulate_grad_batches=2, inference_mode=False, gradient_clip_val=1.0,
        logger=CSVLogger(str(args.output), name="logs"),
        callbacks=[StressWeightScheduler(), ModelCheckpoint(
            dirpath=args.output / "checkpoints", every_n_epochs=1, save_top_k=-1)],
        fast_dev_run=args.smoke_test)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if not args.smoke_test:
        module.model.save(args.output / "model")


if __name__ == "__main__":
    main()
