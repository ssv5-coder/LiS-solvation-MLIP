# Reactive M3GNet potential (model 2.0)

The model used in the paper is the epoch-240 checkpoint of the seed-0 full-network continuation from the original production potential. Its use in the paper was limited to additional constrained scans and continuation simulations. The original production model and its training procedure are in the [parent training guide](../README.md).

## Training data

Extract the Training and Testing archives from the repository with Git LFS. The selection file records the archive path and frame number for 4,000 fitting frames (2,000 of each electrolyte) and 800 historical validation frames. The latter have prior model exposure. The separate input package contains 177 periodic structures and their PBE-D3 energies and forces in compressed ASE trajectory format. Of 105 sequential S–S scan frames, 75 are assigned to training, 15 to validation (windows 5–7), and 15 to evaluation (windows 11–13). A further 48 development frames enter training and 24 enter validation. The selection file records chemistry and trajectory family for the development frames. Keep the original frame order, whole-window assignments, and stress masks. The preparation command checks the geometry and labels of every frame before enabling training.

## Reproduce the fit

Use Linux, Python 3.10 and one CUDA GPU. The validated reactive trainer uses PyTorch 2.2.0, a matching CUDA-enabled DGL 2.1.0 build, NumPy 1.26.4 and MatGL 1.2.3. This DGL version differs from the original model's inference environment. Install torchdata 0.7.1, ASE and pymatgen as well. Start from a fresh checkout with write access to this directory. Obtain the model 2.0 input package from the authors and unpack it at the repository root. It supplies training/model2/selection.json.gz, training/model2/reactive.traj.gz and the two compressed reference reports in training/model2/baseline/.

```bash
cd training/model2
export DGLBACKEND=pytorch
python prepare_data.py /path/to/extracted_archives
python preflight.py
python train_seed0.py --hours 12
```

Preflight checks batch and isolated predictions, finite-difference forces, energy extensivity, parameter updates and save/reload consistency. It computes the original-model force targets used to retain electrolyte accuracy. The trainer samples 2,000 structures per epoch, draws 25% from the new data, and updates the full potential with Adam at an initial learning rate of 0.0001. The force loss gives reactive polysulfide sulfur fivefold weight; original-frame force retention has weight five. Energy and force terms carry unit weight. The trainer accumulates microbatches of four to an effective batch of 56 and increases the stress weight from 0.01 to 0.1. It saves a resumable checkpoint and a separate best checkpoint in `runs/model2_seed0/`. Use `--resume` when continuing a time-limited run. Wall time and floating-point results depend on hardware and library builds.

## Use the selected checkpoint

The separate model package contains the three model files, runtime adapter and inference contract. Extract it so these files reside in `models/reactive_model_2/`. Use the adapter supplied with this potential to preserve the training-time graph indexing. From this directory:

```python
from pathlib import Path
from runtime_adapter import load_potential
model = load_potential(Path('../../models/reactive_model_2'), device='cuda').eval()
```

The reference surface is non-spin-polarized PBE-D3. Consult SI Section S1 and Table S9 for the dataset partitions and validation errors.
