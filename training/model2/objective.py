"""Group-balanced losses with exact accumulation-window normalization."""
from policy import training_bucket, anchor_frame

def sample_weights(data, ids, new_probability):
    desired = {'new': new_probability, 'PS': (1-new_probability)/2,
               'nonPS': (1-new_probability)/2}
    buckets = [training_bucket(data.records[i]) for i in ids]
    available = sum(desired[b] for b in set(buckets))
    return [desired[b] / available / buckets.count(b) for b in buckets]

def components(data, ids, predictions, targets, counts, weights,
               stress_denominator, anchor_denominator):
    import torch
    energy, forces, stress = predictions[:3]
    et, ft, st = targets
    energy = energy.reshape(-1)
    stress = stress.reshape_as(st)
    zero = energy.sum() * 0
    output = {name: zero for name in ['energy', 'force', 'stress', 'anchor']}
    output['energy'] = sum(w * ((energy[j] - et[j]) / counts[j]).square()
                           for j, w in enumerate(weights))
    offset = 0
    for j, (i, w) in enumerate(zip(ids, weights)):
        n = len(data.atoms[i])
        coeff = torch.as_tensor(data.coefficients[i], device=forces.device,
                                dtype=forces.dtype).reshape(-1, 1)
        predicted = forces[offset:offset+n]
        delta = predicted - ft[offset:offset+n]
        output['force'] = output['force'] + w * (coeff * delta.square()).sum()
        if anchor_frame(data.records[i]) and anchor_denominator > 0:
            if i not in data.teacher:
                raise RuntimeError('Missing original-force teacher cache for frame %s' % i)
            teacher = torch.as_tensor(data.teacher[i], device=forces.device, dtype=forces.dtype)
            output['anchor'] = output['anchor'] + w / anchor_denominator * (
                coeff * (predicted-teacher).square()).sum()
        mask = torch.isfinite(st[j])
        if mask.any() and stress_denominator > 0:
            output['stress'] = output['stress'] + w / stress_denominator * (
                stress[j][mask]-st[j][mask]).square().mean()
        offset += n
    return output

def configure_parameters(potential, mode):
    for p in potential.parameters():
        p.requires_grad_(False)
    if mode == 'all':
        for p in potential.model.parameters():
            p.requires_grad_(True)
    elif mode == 'last':
        network = potential.model
        for module in [network.graph_layers[-1], network.three_body_interactions[-1],
                       network.final_layer]:
            for p in module.parameters():
                p.requires_grad_(True)
    elif mode == 'readout':
        for p in potential.model.final_layer.parameters():p.requires_grad_(True)
    else:
        raise ValueError(mode)
    names = [n for n, p in potential.named_parameters() if p.requires_grad]
    if not names or any(n.startswith('element_refs') for n in names):
        raise RuntimeError('Invalid trainable-parameter selection')
    return names
