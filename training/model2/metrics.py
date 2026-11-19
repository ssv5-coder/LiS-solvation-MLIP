from common import *
from policy import source_bucket
import math

def group_name(record):
    if record['partition'] == 'retention':
        return 'historical_' + record['chemistry']
    if record['origin'] == 'new':
        return 'scan_' + record['partition']
    return record['origin'] + '_' + record['partition'] + '_' + source_bucket(record)

def summarize(rows):
    import numpy as np
    result = {}
    for group in sorted({r['group'] for r in rows}):
        rr = [r for r in rows if r['group'] == group]
        force = {}
        for label in sorted({k for r in rr for k in r['metrics']}):
            values = [r['metrics'][label] for r in rr if label in r['metrics']]
            n = sum(v['components'] for v in values)
            force[label] = dict(MAE=sum(v['abs_sum'] for v in values)/n,
                RMSE=math.sqrt(sum(v['square_sum'] for v in values)/n), components=n,
                max_abs=max(v['max_abs'] for v in values) if all('max_abs' in v for v in values) else None)
        relative,relative_rmse = {},{}
        for composition in sorted({r['composition'] for r in rr}):
            errors = np.array([r['energy_error_eV'] for r in rr if r['composition'] == composition])
            if len(errors) > 1:
                delta = abs(errors[:, None]-errors[None, :])
                relative[composition] = float(delta[np.triu_indices(len(errors), 1)].mean())
                relative_rmse[composition]=float(np.sqrt(np.mean(delta[np.triu_indices(len(errors),1)]**2)))
        energies=np.array([r['energy_error_eV_atom'] for r in rr])
        result[group] = dict(task=rr[0]['task'], frames=len(rr), force=force,
            energy_MAE_eV_atom=float(np.mean(abs(energies))),
            energy_RMSE_eV_atom=float(np.sqrt(np.mean(energies**2))),
            energy_bias_eV_atom=float(np.mean(energies)),energy_max_abs_eV_atom=float(np.max(abs(energies))),
            relative_energy_pairwise_MAE_eV=relative,relative_energy_pairwise_RMSE_eV=relative_rmse)
        projected = [abs(r['projected_force_error_eV_A']) for r in rr
                     if 'projected_force_error_eV_A' in r]
        if projected:
            result[group]['projected_force_MAE_eV_A'] = float(np.mean(projected))
            result[group]['projected_force_RMSE_eV_A'] = float(np.sqrt(np.mean(np.array(projected)**2)))
    return result

def historical_baseline(data):
    report = json.loads((HERE / 'baseline/validation.json').read_text())
    rows = []
    for old in report['rows']:
        row = dict(old)
        r = data.records[row['index']]
        row.update(group=group_name(r), task=source_bucket(r), composition=r['composition'])
        rows.append(row)
    return dict(rows=rows, summary=summarize(rows))

def evaluate(model, data, ids, batch_size=4, save_components=False):
    import numpy as np, torch
    model.eval()
    rows, arrays = [], {}
    for start in range(0, len(ids), batch_size):
        chosen = ids[start:start+batch_size]
        graph, lattice, state, targets, counts = data.batch(chosen)
        with torch.enable_grad():
            ep, fp, *_ = model(g=graph, lat=lattice, state_attr=state)
        energies = ep.detach().cpu().numpy().reshape(-1)
        forces = fp.detach().cpu().numpy()
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise RuntimeError('Nonfinite evaluation')
        offset = 0
        for j, i in enumerate(chosen):
            a, r = data.atoms[i], data.records[i]
            predicted = forces[offset:offset+len(a)]
            delta = predicted-data.targets[i][1]
            offset += len(a)
            row = dict(index=i, group=group_name(r), task=source_bucket(r),
                origin=r['origin'], partition=r['partition'], composition=r['composition'],
                family=r.get('family', r.get('group')), window=r.get('window'),
                energy_error_eV=float(energies[j]-data.targets[i][0]),
                energy_error_eV_atom=float((energies[j]-data.targets[i][0])/len(a)), metrics={})
            masks = {'all': list(range(len(a))), 'polysulfide_S': r['polysulfide_indices'],
                     'TFSI_S': r['TFSI_sulfur_indices'], 'Li': np.flatnonzero(a.numbers == 3),
                     'solvent': r['solvent_indices']}
            for element in sorted(set(a.get_chemical_symbols())):
                masks['element_'+element] = [k for k, s in enumerate(a.get_chemical_symbols()) if s == element]
            if r['polysulfide_indices']:
                distances = a.get_all_distances(mic=True)
                masks['neighbors_3p5A'] = [k for k in range(len(a)) if k not in r['polysulfide_indices']
                    and min(distances[k, r['polysulfide_indices']]) <= 3.5]
            if 'pair_indices' in r:
                ii, jj = r['pair_indices']
                vec = a.get_distance(ii, jj, mic=True, vector=True)
                distance = float(np.linalg.norm(vec))
                row.update(S_S_distance_A=distance,
                    projected_force_error_eV_A=float(.5 * np.dot(vec/distance, delta[jj]-delta[ii])))
                masks['reacting_pair'] = [ii, jj]
            for name, indices in masks.items():
                if not len(indices):
                    continue
                v = delta[indices]
                row['metrics'][name] = dict(abs_sum=float(abs(v).sum()),
                    square_sum=float((v*v).sum()), components=int(v.size),
                    max_abs=float(abs(v).max()), MAE=float(abs(v).mean()),
                    RMSE=float(np.sqrt((v*v).mean())))
            rows.append(row)
            if save_components:
                arrays[str(i)] = predicted.copy()
    return dict(rows=rows, summary=summarize(rows)), arrays
