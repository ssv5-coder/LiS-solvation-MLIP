"""Raw labels, stable atom identities, and reusable MatGL graphs."""
from common import *
from policy import source_bucket, training_bucket, force_coefficients, check_split

class Dataset:
    def __init__(self, reactive_s_weight=5.):
        import numpy as np
        from ase.io.trajectory import Trajectory
        from ase.units import GPa
        from matgl.apps.pes import Potential
        from matgl.ext.ase import Atoms2Graph
        require_gate('dataset')
        self.manifest = json.loads((HERE / 'dataset/manifest.json').read_text())
        self.records = self.manifest['records']
        self.atoms = list(Trajectory(HERE / 'dataset/labels.traj'))
        assert len(self.records) == len(self.atoms)
        check_split(self.records)
        potential = Potential.load(MODEL)
        converter = Atoms2Graph(potential.model.element_types, potential.model.cutoff)
        self.graphs, self.targets, self.coefficients = [], [], []
        for i, (atoms, record) in enumerate(zip(self.atoms, self.records)):
            assert record['index'] == i
            assert atoms.pbc.all()
            assert all(atoms.numbers[j] == 16 for j in record['polysulfide_indices'])
            graph, lattice, state = converter.get_graph(atoms)
            self.graphs.append((graph, lattice, np.asarray(state)))
            try:
                stress = atoms.get_stress(voigt=False) / GPa
            except Exception:
                stress = np.full((3, 3), np.nan)
            self.targets.append((float(atoms.get_potential_energy()), atoms.get_forces(), stress))
            self.coefficients.append(force_coefficients(atoms.get_chemical_symbols(),
                record['polysulfide_indices'], source_bucket(record) == 'breaking', reactive_s_weight))
            record['composition'] = atoms.get_chemical_formula()
        self.teacher = {}
        path = HERE / 'teacher/forces.npz'
        if path.exists():
            meta = json.loads((path.parent / 'manifest.json').read_text())
            assert meta['baseline_hashes'] == verify_model()
            assert meta['adapter_sha256'] == sha(HERE / 'runtime_adapter.py')
            for i, digest in meta['geometry_hashes'].items():
                assert self.records[int(i)]['geometry_hash'] == digest
            with np.load(path) as values:
                self.teacher = {int(k): values[k].copy() for k in values.files}

    def ids(self, partition, origin=None):
        return [i for i, r in enumerate(self.records) if r['partition'] == partition
                and (origin is None or r['origin'] == origin)]

    def validation_ids(self):
        return self.ids('validation') + self.ids('retention')

    def training_panel(self, exclude_active=False):
        import numpy as np
        ids = self.ids('train')
        if exclude_active:ids=[i for i in ids if self.records[i]['origin']!='active']
        result = [i for i in ids if self.records[i]['origin'] != 'original']
        for chemistry in ['PS', 'nonPS']:
            pool = [i for i in ids if training_bucket(self.records[i]) == chemistry]
            result += [pool[j] for j in np.linspace(0, len(pool)-1, min(128, len(pool)), dtype=int)]
        return sorted(set(result))

    def tiny_ids(self):
        import numpy as np
        ids = self.ids('train')
        chosen = []
        for chemistry in ['PS', 'nonPS']:
            pool = [i for i in ids if source_bucket(self.records[i]) == chemistry]
            chosen += [pool[0], pool[len(pool)//2]]
        pool = sorted([i for i in ids if source_bucket(self.records[i]) == 'breaking'],
                      key=lambda i: self.records[i]['S_S_distance_A'])
        chosen += [pool[j] for j in np.linspace(0, len(pool)-1, 4, dtype=int)]
        assert len(set(chosen)) == 8
        return chosen

    def batch(self, indices, device='cuda', dtype=None):
        import numpy as np, torch, dgl
        dtype = torch.float32 if dtype is None else dtype
        graphs = [self.graphs[i] for i in indices]
        # MatGL mutates graph fields; local_var isolates those fields between forwards.
        graph = dgl.batch([x[0].local_var() for x in graphs]).to(device)
        lattice = torch.cat([x[1] for x in graphs]).to(device=device, dtype=dtype)
        state = torch.as_tensor(np.stack([x[2] for x in graphs]), device=device, dtype=dtype)
        tensor = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        counts = tensor([len(self.atoms[i]) for i in indices])
        targets = (tensor([self.targets[i][0] for i in indices]),
                   tensor(np.concatenate([self.targets[i][1] for i in indices])),
                   tensor(np.stack([self.targets[i][2] for i in indices])))
        return graph, lattice, state, targets, counts

    def training_draws(self, rng, count, new_probability, exclude_active=False):
        import numpy as np
        ids = self.ids('train')
        if exclude_active:ids=[i for i in ids if self.records[i]['origin']!='active']
        pools = {name: [i for i in ids if training_bucket(self.records[i]) == name]
                 for name in ['PS', 'nonPS', 'new']}
        if not all(pools.values()):
            raise RuntimeError('A training chemistry is missing')
        weights = {'PS': (1-new_probability)/2, 'nonPS': (1-new_probability)/2,
                   'new': new_probability}
        out = []
        for name in rng.choice(list(weights), count, p=list(weights.values())):
            pool = pools[name]
            if name == 'new':
                # Equal family/window sampling, then equal sampling within each group.
                groups = {}
                for i in pool:
                    r = self.records[i]
                    key = r.get('family', 'scan_%s' % r.get('window'))
                    groups.setdefault(key, []).append(i)
                group = list(groups)[int(rng.integers(len(groups)))]
                pool = groups[group]
            out.append(int(rng.choice(pool)))
        return np.asarray(out)
