"""Pure selection and allocation policy, shared by tests and remote workers."""
import itertools, math

def variants():
    return [dict(name='screen_%02d' % i, learning_rate=lr, energy_weight=ew,
                 anchor_weight=aw, layers=layers, new_probability=.5, seed=0)
            for i, (lr, ew, aw, layers) in enumerate(itertools.product(
                [1e-5, 3e-5], [.01, .1], [0., .5], ['all', 'last']))]

def source_bucket(record):
    return record.get('task', 'breaking' if record['origin'] == 'new' else record['chemistry'])

def training_bucket(record):
    """New equilibrium labels belong to new-data sampling, not dilute replay."""
    return 'new' if record['origin'] in ['new', 'active'] else record['chemistry']

def anchor_frame(record):
    return record['origin'] == 'original' and record.get('partition', 'train') == 'train'

def atomic_groups(symbols, polysulfide_indices):
    ps = set(polysulfide_indices)
    groups = {}
    for i, element in enumerate(symbols):
        name = 'polysulfide_S' if i in ps else ('TFSI_S' if element == 'S' else element)
        groups.setdefault(name, []).append(i)
    return groups

def force_coefficients(symbols, ps_indices, reactive, sulfur_weight=5.):
    """Coefficients over force components sum to one for each structure."""
    groups = atomic_groups(symbols, ps_indices)
    if not math.isfinite(sulfur_weight) or sulfur_weight<=0:raise ValueError('Invalid sulfur weight')
    weights = {g: sulfur_weight if reactive and g == 'polysulfide_S' else 1. for g in groups}
    total = sum(weights.values())
    result = [0.] * len(symbols)
    for group, indices in groups.items():
        for i in indices:
            result[i] = weights[group] / total / (3 * len(indices))
    return result

def acceptance(summary, baseline, thresholds, expected=None):
    """All evaluated groups are mandatory; a missing reference is never a pass."""
    ratios, failures = {}, []
    for group in (expected if expected is not None else baseline):
        if group not in summary or group not in baseline:
            failures.append(group + '/missing')
            continue
        c, b = summary[group], baseline[group]
        breaking = c['task'] == 'breaking'
        limit = thresholds['breaking_energy_MAE_eV_atom_max' if breaking else 'original_energy_MAE_eV_atom_max']
        ratios[group + '/energy'] = c['energy_MAE_eV_atom'] / limit
        if breaking:
            key = group + '/polysulfide_S'
            value = c['force']['polysulfide_S']['MAE']
            ratios[key] = value / thresholds['sulfur_MAE_eV_A_strict_max']
            if value >= thresholds['sulfur_MAE_eV_A_strict_max']:
                failures.append(key)
            if 'projected_force_MAE_eV_A' not in c or 'projected_force_MAE_eV_A' not in b:
                failures.append(group + '/missing_projected')
            else:
                ratios[group + '/projected'] = c['projected_force_MAE_eV_A'] / max(
                    1e-12, b['projected_force_MAE_eV_A'] * thresholds['projected_ratio_max'])
        else:
            for label, value in b['force'].items():
                if label != 'all' and not label.startswith('element_'):
                    continue
                if label not in c['force']:
                    failures.append(group + '/' + label + '/missing')
                else:
                    ratios[group + '/' + label] = c['force'][label]['MAE'] / max(
                        1e-12, value['MAE'] * thresholds['original_force_ratio_max'])
        for composition, value in b['relative_energy_pairwise_MAE_eV'].items():
            if composition not in c['relative_energy_pairwise_MAE_eV']:
                failures.append(group + '/relative_missing/' + composition)
            else:
                ratios[group + '/relative/' + composition] = c['relative_energy_pairwise_MAE_eV'][composition] / max(
                    1e-12, value * thresholds['relative_energy_ratio_max'])
    failures += [k for k, v in ratios.items() if not math.isfinite(v) or v > 1.0]
    sulfur = [v['force']['polysulfide_S']['MAE'] for v in summary.values() if v['task'] == 'breaking']
    projected = [v['projected_force_MAE_eV_A'] for v in summary.values()
                 if v['task'] == 'breaking' and 'projected_force_MAE_eV_A' in v]
    eligible = not failures and bool(ratios) and bool(sulfur)
    worst = max(ratios.values(), default=1e30)
    rank = [not eligible, worst if not eligible else 0., max(sulfur, default=1e30),
            max(projected, default=1e30)]
    return dict(passed=eligible, failures=sorted(set(failures)), ratios=ratios, rank=rank,
                progress_score=math.log1p(worst) + math.log1p(max(sulfur, default=1e30) / .1))

def check_budget(entries, candidate, config):
    entries = [x for x in entries if x.get('state') != 'not_started'] + [candidate]
    pm = [x for x in entries if x['kind'].startswith('pm')]
    stages = {x['kind'] for x in pm}
    if not stages.issubset({'pm1', 'pm2'}) or len(stages) > config['pm_sessions']:
        raise RuntimeError('Two PM stage budgets already reserved')
    for x in pm:
        if x['gpus'] != 16 or x['seconds'] > config['pm_seconds_per_session']:
            raise RuntimeError('PM allocation exceeds authorization')
    for stage in stages:
        if sum(x.get('actual_seconds', x['seconds']) for x in pm if x['kind'] == stage) > config['pm_seconds_per_session']:
            raise RuntimeError('PM stage runtime budget exceeded, including prior attempts')
    extra = [x for x in entries if not x['kind'].startswith('pm')]
    if any(x['gpus'] != 1 for x in extra):
        raise RuntimeError('Final training must use one GPU')
    used = lambda x: x.get('actual_seconds', x['seconds'])
    if sum(used(x) for x in extra) > config['ampere_seconds_total']:
        raise RuntimeError('Additional 48-hour budget exceeded')
    for seed in [0, 1, 2]:
        if sum(used(x) for x in extra if x.get('seed') == seed) > config['long_seconds_per_seed']:
            raise RuntimeError('Seed training allowance exceeded')
    if sum(used(x) for x in extra if x['kind'] == 'final') > config['final_check_seconds']:
        raise RuntimeError('Final-check allowance exceeded')
    if sum(x['gpus'] * used(x) for x in entries) > config['total_gpu_hours'] * 3600:
        raise RuntimeError('Total 176 GPU-hour budget exceeded')

def check_split(records):
    families, hashes = {}, {}
    for r in records:
        if r.get('origin') != 'active':
            continue
        for value, seen in [(r['family'], families), (r['geometry_hash'], hashes)]:
            if value in seen and seen[value] != r['partition']:
                raise RuntimeError('New-data leakage between partitions')
            seen[value] = r['partition']
    return True
