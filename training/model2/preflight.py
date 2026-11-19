"""Scheduled GPU checks required before any diagnostic or candidate training."""
from common import *
import traceback

def main():
    import copy, importlib.metadata as metadata
    import numpy as np, torch
    from matgl.ext.ase import Atoms2Graph
    from dataset import Dataset
    from objective import components, sample_weights, configure_parameters
    from metrics import evaluate, historical_baseline
    from runtime_adapter import load_potential, save_potential, install
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    versions = {n: metadata.version(n) for n in ['torch', 'dgl', 'matgl', 'numpy', 'ase']}
    assert versions['matgl'] == '1.2.3' and versions['dgl'].startswith('2.1.0')
    assert versions['numpy'] == '1.26.4' and versions['torch'].startswith('2.2.0')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0)
    data = Dataset()
    ids = data.tiny_ids()
    assert len(data.ids('train', 'new')) == 75
    assert all(data.records[i]['window'] not in [5, 6, 7, 11, 12, 13]
               for i in data.ids('train', 'new'))
    model = load_potential(MODEL).eval()
    baseline = historical_baseline(data)
    save(HERE / 'checks/historical_reference.json', baseline)
    report, cached = evaluate(model, data, ids, save_components=True)
    data.teacher.update(cached)
    checks = {}
    # Compare the same structures alone and in a mixed-size batch.
    single, sf = evaluate(model, data, ids, batch_size=1, save_components=True)
    checks['batch_force_max'] = max(float(abs(cached[str(i)]-sf[str(i)]).max()) for i in ids)
    assert checks['batch_force_max'] < .002
    # The teacher cache is indexed by integer frame IDs.
    data.teacher = {int(k): v for k, v in data.teacher.items()}
    for mode in ['all', 'last', 'readout']:
        smoke = load_potential(MODEL).train()
        names = configure_parameters(smoke, mode)
        before = {n: p.detach().clone() for n, p in smoke.named_parameters()}
        optimizer = torch.optim.Adam([p for p in smoke.parameters() if p.requires_grad], lr=1e-5)
        weights = sample_weights(data, ids, .5)
        sd = sum(w for i, w in zip(ids, weights) if np.isfinite(data.targets[i][2]).any())
        ad = sum(w for i, w in zip(ids, weights) if data.records[i]['task'] != 'breaking')
        optimizer.zero_grad()
        for offset in range(0, len(ids), 4):
            chosen = ids[offset:offset+4]
            g, lat, state, targets, counts = data.batch(chosen)
            prediction = smoke(g=g, lat=lat, state_attr=state)
            parts = components(data, chosen, prediction, targets, counts,
                               weights[offset:offset+4], sd, ad)
            loss = .01*parts['energy'] + parts['force'] + .01*parts['stress'] + .5*parts['anchor']
            assert torch.isfinite(loss)
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in smoke.parameters() if p.requires_grad], 1.)
        assert torch.isfinite(norm)
        optimizer.step()
        changed = [n for n, p in smoke.named_parameters() if not torch.equal(before[n], p)]
        assert changed and all(n in names for n in changed)
        assert all(not n.startswith('element_refs') for n in changed)
        checks[mode+'_updated_parameters'] = len(changed)
        del smoke, before, optimizer
    # Numerical derivatives and extensivity in float64 resolve energy cancellation.
    double = copy.deepcopy(model).double()
    install(torch.float64)
    converter = Atoms2Graph(double.model.element_types, double.model.cutoff)
    def predict(atoms, potential=double):
        graph, lattice, state = converter.get_graph(atoms)
        graph = graph.to('cuda')
        graph.ndata['frac_coords'] = graph.ndata['frac_coords'].double()
        lattice = lattice.to(device='cuda', dtype=torch.float64)
        attributes = torch.as_tensor(np.asarray(state), device='cuda', dtype=torch.float64)
        energy, force, *_ = potential(g=graph, lat=lattice, state_attr=attributes)
        return float(energy.detach().cpu().reshape(-1)[0]), force.detach().cpu().numpy()
    atoms = data.atoms[ids[-1]].copy()
    e0, f0 = predict(atoms)
    derivatives = []
    for i in data.records[ids[-1]]['polysulfide_indices'][:2] + [int(np.flatnonzero(atoms.numbers == 8)[0])]:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[i, 0] += 1e-4
        minus.positions[i, 0] -= 1e-4
        derivatives.append(abs(-(predict(plus)[0]-predict(minus)[0])/2e-4-f0[i, 0]))
    er, fr = predict(atoms.repeat((2, 1, 1)))
    checks.update(finite_difference_max_eV_A=float(max(derivatives)),
                  extensivity_eV_atom=abs(er-2*e0)/(2*len(atoms)),
                  repeat_force_max_eV_A=float(abs(fr-np.tile(f0, (2, 1))).max()))
    assert checks['finite_difference_max_eV_A'] < .003
    assert checks['extensivity_eV_atom'] < 1e-5 and checks['repeat_force_max_eV_A'] < .002
    install(torch.float32)
    del double
    export = HERE / 'checks/roundtrip'
    save_potential(model, export)
    restored = load_potential(export)
    _, restored_forces = evaluate(restored, data, ids, save_components=True)
    checks['reload_force_max_eV_A'] = max(float(abs(cached[str(i)]-restored_forces[str(i)]).max()) for i in ids)
    assert checks['reload_force_max_eV_A'] < .002
    del restored
    # Teacher predictions are from the corrected baseline, never its erroneous energies.
    teacher_ids = [i for i in data.ids('train') if data.records[i]['origin'] == 'original']
    missing = [i for i in teacher_ids if i not in data.teacher]
    if missing:
        _, teacher = evaluate(model, data, missing, save_components=True)
        data.teacher.update({int(k): v for k, v in teacher.items()})
    (HERE / 'teacher').mkdir(exist_ok=True)
    np.savez_compressed(HERE / 'teacher/forces.npz', **{str(i): data.teacher[i] for i in teacher_ids})
    save(HERE / 'teacher/manifest.json', dict(baseline_hashes=verify_model(),
        adapter_sha256=sha(HERE / 'runtime_adapter.py'), frames=len(teacher_ids),
        geometry_hashes={str(i): data.records[i]['geometry_hash'] for i in teacher_ids},
        energy_targets_anchored=False))
    result = record_gate('ready', checks=checks, versions=versions,
        teacher_frames=len(teacher_ids), job=os.environ.get('SLURM_JOB_ID'),
        baseline_name=CONFIG['baseline_name'])
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(HERE / 'checks/preflight_failure.json',
             dict(error=str(exc), traceback=traceback.format_exc(), timestamp=time.time()))
        raise
