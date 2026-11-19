"""Rebuild the model 2.0 dataset from the extracted public AIMD archives."""
import argparse
import gzip
import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.io.trajectory import Trajectory

from common import HERE, save, sha, verify_model
from policy import check_split


def label_hash(atoms):
    digest = hashlib.sha256()
    fields = [atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc,
              [atoms.get_potential_energy()], atoms.get_forces()]
    try:
        fields.append(atoms.get_stress())
    except (RuntimeError, NotImplementedError):
        pass
    for field in fields:
        digest.update(np.asarray(field, dtype='<f8').round(8).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path, help='Contains Training/ and Testing/.')
    args = parser.parse_args()
    verify_model()
    records = json.loads(gzip.decompress((HERE / 'selection.json.gz').read_bytes()))
    check_split(records)
    with TemporaryDirectory(dir=HERE) as tmp, ExitStack() as stack:
        tmp = Path(tmp)
        reactive = tmp / 'reactive.traj'
        reactive.write_bytes(gzip.decompress((HERE / 'reactive.traj.gz').read_bytes()))
        sources = {}
        with Trajectory(tmp / 'labels.traj', 'w') as target:
            for i, record in enumerate(records):
                path = reactive if record['origin'] != 'original' else args.data_root / record['source']
                if path not in sources:
                    sources[path] = stack.enter_context(Trajectory(path))
                atoms = sources[path][record['source_index']]
                if record['index'] != i or label_hash(atoms) != record['label_sha256']:
                    raise ValueError(f'Frame {i}: geometry or DFT labels differ from the archived training input')
                target.write(atoms)
        folder = HERE / 'dataset'
        folder.mkdir(exist_ok=True)
        (tmp / 'labels.traj').replace(folder / 'labels.traj')
        save(folder / 'manifest.json', {'records': records})
    for filename in ('validation.json', 'active_validation.json'):
        path = HERE / 'baseline' / filename
        with gzip.open(str(path) + '.gz', 'rb') as source, path.open('wb') as target:
            shutil.copyfileobj(source, target)
    deps = {f'dataset/{n}': sha(folder / n) for n in ('manifest.json', 'labels.traj')}
    save(HERE / 'gates/dataset.json', {'passed': True, 'dependencies': deps})
    print(f'Verified {len(records)} periodic configurations.')


if __name__ == '__main__':
    main()
