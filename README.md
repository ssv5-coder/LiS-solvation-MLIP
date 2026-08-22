# LiS-solvation-MLIP

Fine-tuned M3GNet interatomic potential and DFT reference data accompanying:

**Machine-Learned Molecular Dynamics Reveals Solvent-Coupled Polysulfide
Dissociation in Lithium-Sulfur Electrolytes**

The model was developed for LiTFSI in mixed DME/DOL solvent with and without
lithium polysulfides. It predicts energies, atomic forces, and stresses and was
used for the large-scale ML-MD simulations reported in the manuscript.

## Contents

```text
LiS-solvation-MLIP/
|-- data/
|   |-- Training.tar.gz
|   |-- Testing.tar.gz
|   `-- production_trajectories/
|       `-- README.md
|-- examples/
|   `-- input.extxyz
|-- models/
|   `-- fine_tuned_m3gnet_model/
|       |-- model.json
|       |-- model.pt
|       `-- state.pt
|-- CITATION.cff
|-- LICENSE
|-- README.md
|-- example_ase.py
`-- requirements.txt
```

The DFT archives are stored with Git LFS. Install Git LFS before cloning if you
need these files:

```bash
git lfs install
git clone https://github.com/ssv5-coder/LiS-solvation-MLIP.git
cd LiS-solvation-MLIP
git lfs pull
```

## Scientific scope

The model was fine-tuned on PBE-D3 AIMD configurations of LiTFSI/DME/DOL
electrolytes with and without Li2S6. Independent DFT tests additionally sampled
changes in solvent ratio and Li2Sx chain length. This validation does not imply
transferability to arbitrary salts, solvents, electrode interfaces,
decomposition products, temperatures, or reactive pathways. Extrapolative use
should be checked against suitable first-principles calculations.

## Installation

Python 3.10 is recommended. Create a clean environment and install the pinned
DGL-era MatGL stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checkpoint was serialized with MatGL 1.2.3. MatGL 3 and later use a
different backend and cannot load this historical checkpoint without a
validated model conversion.

## ASE example

Evaluate the included structure:

```bash
python example_ase.py examples/input.extxyz
```

Or provide any ASE-readable periodic structure containing only chemical
environments within the model's validated scope:

```bash
python example_ase.py POSCAR
```

The script reports total energy in eV, forces in eV/Angstrom, and stress in ASE
Voigt order in eV/Angstrom^3. The stress conversion used by the released script
matches the validated manuscript benchmark.

The model can also be attached to standard ASE optimizers or molecular-dynamics
classes. The manuscript production simulations, however, used LAMMPS and the
specific equilibration and thermostat protocol reported in the paper; the
single-point script is a deployment example, not a reproduction of that full
workflow.

## Data

- `Training.tar.gz` contains the AIMD configurations used for model fitting.
- `Testing.tar.gz` contains the chemistry-specific held-out AIMD trajectories.
- `production_trajectories/` documents the final PS and non-PS 300 K coordinate
  trajectories used by the production analyses. The trajectory binaries are
  not included in the current release and will be distributed separately.

Preserve the published data split when reproducing model metrics. Additional
trajectory details, restart caveats, and frame counts are given in
[`data/production_trajectories/README.md`](data/production_trajectories/README.md).

## Model architecture

The authoritative serialized metadata are in
`models/fine_tuned_m3gnet_model/model.json`. The checkpoint uses three M3GNet
interaction blocks, 64-dimensional node and edge embeddings, a 5.0 Angstrom
pair cutoff, a 4.0 Angstrom three-body cutoff, and energy/force/stress outputs.

## Citation

Please cite the accompanying manuscript and the original M3GNet and MatGL
software papers. Bibliographic metadata will be updated after publication.

## License

The repository is released under the [MIT License](LICENSE). Third-party
packages retain their respective licenses.
