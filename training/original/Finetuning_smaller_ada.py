#!/usr/bin/env python
from __future__ import annotations

import os
import shutil
import warnings
import math
import random
import concurrent.futures

import numpy as np
import torch
import lightning as pl
from functools import partial
from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS

# Suppress warnings for clearer output
warnings.simplefilter("ignore")
torch.manual_seed(0)
torch.cuda.empty_cache()

from ase.units import GPa
import pymatgen.io.ase

# -------------------------------
# Custom Callback to schedule stress weight
# -------------------------------
from lightning.pytorch.callbacks import Callback

class StressWeightScheduler(Callback):
    """
    Callback that linearly increases the stress weight from an initial value to a final value 
    by a specified plateau epoch, then holds it constant for the remainder of training.
    """
    def __init__(self, initial=0.01, final=0.1, plateau_epoch=20, total_epochs=20):
        super().__init__()
        self.initial = initial
        self.final = final
        self.plateau_epoch = plateau_epoch
        self.total_epochs = total_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # If the current epoch is less than the plateau, interpolate linearly.
        if epoch < self.plateau_epoch:
            new_weight = self.initial + (self.final - self.initial) * (epoch / max(self.plateau_epoch - 1, 1))
        else:
            new_weight = self.final
        pl_module.stress_weight = new_weight
        # Log the stress weight so it appears in your CSV/TensorBoard logs.
        pl_module.log("stress_weight", new_weight, prog_bar=True, on_epoch=True)
        print(f"Epoch {epoch+1}: Updated stress weight to {new_weight:.4f}")

# -------------------------------
# Data and Model Preparation
# -------------------------------
element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

# Create training and validation datasets
dataset = MGLDataset(
    save_dir='./Data/save_train_best',
    threebody_cutoff=4.0, 
    include_line_graph=True
)

dataset_val = MGLDataset(
    save_dir='./Data/save_val_best',
    threebody_cutoff=4.0,
    include_line_graph=True
)

# Split datasets (using 100% of training data here)
train_data, _, _ = split_dataset(
    dataset,
    frac_list=[1, 0, 0.0],
    shuffle=True,
    random_state=42,
)

_, val_data, _ = split_dataset(
    dataset_val,
    frac_list=[0, 1, 0],
    shuffle=True,
    random_state=42,
)

# Setup DataLoaders
my_collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)
train_loader, val_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    collate_fn=my_collate_fn,
    batch_size=28,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device='cpu')
)

# Load pre-trained model
model_save_path = "./big_boy_m3gnet_smaller/"
m3gnet = matgl.load_model(path=model_save_path)
model_pretrained = m3gnet.model
property_offset = m3gnet.element_refs.property_offset

# Instantiate the fine-tuning module with stress_weight starting at 0.0
lit_module_finetune = PotentialLightningModule(
    model=model_pretrained,
    element_refs=property_offset,
    lr=1e-4,
    include_line_graph=True,
    stress_weight=0.01,   # Starting stress weight is 0.0
    force_weight=1.0,
    energy_weight=1.0,
    decay_steps=50,
    decay_alpha=0.1
)

# -------------------------------
# Trainer Setup and ModelCheckpoint Callback
# -------------------------------
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="/fs/ddn/sdf/group/ml/MLIP_Li_S_Fl_Elec/AIMD/chgnet/m3gnet_with_stress/model_chkpts_smaller_stress_ada_2_second",
    filename="m3gnet_smaller_0.0_{epoch:02d}-{val_loss:.6f}",
    every_n_epochs=1,
    save_top_k=-1,
)

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()
logger = CSVLogger("logs", name="M3GNet_finetuning_smaller")

# Instantiate the Trainer with the StressWeightScheduler callback.
# The callback will linearly ramp the stress weight from 0.0 to 0.1 until epoch 20,
# then hold it at 0.1 for the remainder of the training (up to 50 epochs).
trainer = pl.Trainer(
    accumulate_grad_batches=2,
    callbacks=[checkpoint_callback, StressWeightScheduler(initial=0.01, final=0.1, plateau_epoch=20, total_epochs=20)],
    accelerator='gpu',
    devices=1,
    max_epochs=20,
    logger=logger,
    inference_mode=False,
    gradient_clip_val=1.0,
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# -------------------------------
# Train and Save the Model
# -------------------------------
trainer.fit(
    model=lit_module_finetune,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)

model_save_path = "./big_boy_m3gnet_smaller_stress_ada_2_best/"
lit_module_finetune.model.save(model_save_path)
trained_model = matgl.load_model(path=model_save_path)
