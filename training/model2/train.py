"""One-GPU training with exact resumable optimizer boundaries and grouped metrics."""
from common import *
from policy import source_bucket, anchor_frame, acceptance
import argparse, signal, traceback, math, subprocess

def gpu_observation(torch):
    result=dict(visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        allocated_memory_bytes=torch.cuda.memory_allocated(),peak_memory_bytes=torch.cuda.max_memory_allocated())
    try:
        result['node_gpu_utilization_csv']=subprocess.check_output(['nvidia-smi',
            '--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
            universal_newlines=True,timeout=5).strip()
    except Exception as exc:result['utilization_unavailable']=str(exc)
    return result

def atomic_checkpoint(path, state):
    import torch
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)

def train(config, deadline, resume=None):
    import numpy as np, torch
    from dataset import Dataset
    from objective import configure_parameters, components, sample_weights
    from runtime_adapter import load_potential, save_potential
    from metrics import evaluate, historical_baseline
    require_gate('ready')
    verify_model()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    data = Dataset(reactive_s_weight=config.get('reactive_s_weight',5.))
    fingerprint = dependencies()
    model = load_potential(MODEL)
    trainable = configure_parameters(model, config['layers'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
    settings = CONFIG['training']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5,
        patience=settings['lr_patience_epochs']//settings['validation_interval']-1,
        threshold=settings['meaningful_improvement'], threshold_mode='rel',
        min_lr=settings['lr_min'])
    diagnostic = config.get('diagnostic', False)
    output = HERE / 'runs' / config['name']
    output.mkdir(parents=True, exist_ok=True)
    with lock(config['name']):
        if (output / 'manifest.json').exists() and resume is None:
            raise RuntimeError('Existing run requires explicit resume')
        manifest = dict(configuration=config, dependencies=fingerprint, trainable_parameters=trainable,
            baseline_hashes=verify_model(), job=os.environ.get('SLURM_JOB_ID'),
            host=os.uname().nodename, GPU=torch.cuda.get_device_name(),
            initial_seed=config['seed'], production_approved=False)
        if not (output / 'manifest.json').exists():
            save(output / 'manifest.json', manifest)
        else:
            previous = json.loads((output / 'manifest.json').read_text())
            if previous['configuration'] != config or previous['dependencies'] != manifest['dependencies']:
                raise RuntimeError('Resume code/data/configuration differ from checkpoint run')
        state = dict(epoch=0, cursor=0, sample=None, history=[], best_rank=None,
                     best_score=None, last_improved_epoch=0, epoch_parts={}, updates=0,
                     completed_seconds=0.)
        if resume:
            checkpoint = torch.load(resume, map_location='cpu')
            if checkpoint['configuration'] != config or checkpoint['dependencies'] != fingerprint:
                raise RuntimeError('Incompatible restart')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for optimizer_state in optimizer.state.values():
                for k, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[k] = value.to('cuda')
            scheduler.load_state_dict(checkpoint['scheduler'])
            rng.bit_generator.state = checkpoint['numpy_rng']
            torch.set_rng_state(checkpoint['torch_rng'])
            torch.cuda.set_rng_state(checkpoint['cuda_rng'])
            state = checkpoint['progress']
        baseline = historical_baseline(data)
        active_path = HERE / 'baseline/active_validation.json'
        if active_path.exists():
            active = json.loads(active_path.read_text())
            baseline['summary'].update(active['summary'])
        expected = {r for r in baseline['summary']}
        if any(r['origin'] == 'active' and r['partition'] == 'validation' for r in data.records) and not active_path.exists():
            raise RuntimeError('Fresh validation baseline missing')
        validation_ids = data.validation_ids()
        panel = data.tiny_ids() if diagnostic else data.training_panel(config.get('exclude_active_training',False))
        stopped = [False]
        for sig in [signal.SIGTERM, signal.SIGUSR1]:
            signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
        start_time = time.time()
        entry_seconds = state['completed_seconds']

        def checkpoint():
            state['completed_seconds'] = entry_seconds + time.time()-start_time
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), progress=state,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), configuration=config, dependencies=fingerprint)

        def persist(status):
            atomic_checkpoint(output / 'last.pt', checkpoint())
            save(output / 'status.json', dict(state=status, epoch=state['epoch'], cursor=state['cursor'],
                updates=state['updates'], best_rank=state['best_rank'],
                completed_seconds=state['completed_seconds'], production_approved=False,
                job=os.environ.get('SLURM_JOB_ID'), timestamp=time.time()))

        if not state['history'] and not (output / 'initial_training.json').exists():
            initial, _ = evaluate(model, data, panel)
            save(output / 'initial_training.json', initial)
        epoch_limit = 2000 if diagnostic else settings['max_epochs']
        while state['epoch'] < epoch_limit:
            if time.time() >= deadline-90 or stopped[0]:
                persist('checkpointed_time_limit')
                return
            epoch_start = time.time()
            model.train()
            if state['sample'] is None:
                state['sample'] = panel if diagnostic else data.training_draws(
                    rng, settings['draws_per_epoch'], config['new_probability'],
                    config.get('exclude_active_training',False)).tolist()
                state['cursor'] = 0
                state['epoch_parts'] = dict(energy=0., force=0., stress=0., anchor=0.,
                                           steps=0, clipped=0, gradient_norm_sum=0.)
            sample = state['sample']
            effective = len(panel) if diagnostic else settings['effective_batch']
            stress_weight = 0. if diagnostic else settings['stress_initial'] + (
                settings['stress_final']-settings['stress_initial']) * min(
                1., state['epoch']/settings['stress_ramp_epochs'])
            for start in range(state['cursor'], len(sample), effective):
                chunk = sample[start:start+effective]
                weights = sample_weights(data, chunk, config['new_probability'])
                stress_denominator = sum(w for i, w in zip(chunk, weights)
                                         if np.isfinite(data.targets[i][2]).any())
                anchor_denominator = sum(w for i, w in zip(chunk, weights)
                    if anchor_frame(data.records[i])) if config['anchor_weight'] else 0.
                optimizer.zero_grad(set_to_none=True)
                step_parts = dict(energy=0., force=0., stress=0., anchor=0.)
                gradient_diagnostics = {}
                for offset in range(0, len(chunk), settings['microbatch']):
                    ids = chunk[offset:offset+settings['microbatch']]
                    graph, lattice, attributes, targets, counts = data.batch(ids)
                    predictions = model(g=graph, lat=lattice, state_attr=attributes)
                    parts = components(data, ids, predictions, targets, counts,
                        weights[offset:offset+len(ids)], stress_denominator, anchor_denominator)
                    if start == 0 and offset == 0 and state['epoch'] % 5 == 0:
                        for key in ['energy', 'force']:
                            gradients = torch.autograd.grad(parts[key], parameters,
                                retain_graph=True, allow_unused=True)
                            gradient_diagnostics[key] = math.sqrt(sum(float(g.detach().square().sum())
                                for g in gradients if g is not None))
                    loss = config['energy_weight']*parts['energy'] + parts['force'] + (
                        stress_weight*parts['stress']) + config['anchor_weight']*parts['anchor']
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    loss.backward()
                    for name, value in parts.items():
                        step_parts[name] += float(value.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'])
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite parameter gradient')
                optimizer.step()
                state['updates'] += 1
                state['cursor'] = start+len(chunk)
                for name, value in step_parts.items():
                    state['epoch_parts'][name] += value
                state['epoch_parts']['steps'] += 1
                state['epoch_parts']['clipped'] += int(float(norm) > settings['gradient_clip'])
                state['epoch_parts']['gradient_norm_sum'] += float(norm)
                if gradient_diagnostics:
                    save(output / 'gradients_latest.json', dict(epoch=state['epoch'], **gradient_diagnostics))
                if stopped[0] or time.time() >= deadline-60:
                    persist('checkpointed_time_limit')
                    return
            completed_epoch = state['epoch']+1
            # Full train-panel metrics each ordinary epoch; tiny diagnostics every 25 steps.
            train_report = None
            if not diagnostic or completed_epoch % 25 == 0 or completed_epoch == 1:
                train_report, _ = evaluate(model, data, panel)
                save(output / 'training_latest.json', train_report)
            do_validation = diagnostic and train_report is not None or (
                not diagnostic and (completed_epoch % settings['validation_interval'] == 0 or completed_epoch == 1))
            row = dict(epoch=completed_epoch, updates=state['updates'],
                seconds=time.time()-epoch_start, learning_rate=optimizer.param_groups[0]['lr'],
                loss_components={k: v/max(1, state['epoch_parts']['steps'])
                    for k, v in state['epoch_parts'].items() if k not in ['steps']},
                training=train_report['summary'] if train_report else None)
            if not diagnostic and completed_epoch % 5 == 0:
                row['gpu']=gpu_observation(torch)
            report, verdict = None, None
            if do_validation:
                if diagnostic:
                    report = train_report
                    all_force = max(v['force']['all']['MAE'] for v in report['summary'].values())
                    sulfur = max(v['force']['polysulfide_S']['MAE'] for v in report['summary'].values()
                                 if 'polysulfide_S' in v['force'])
                    energy = max(v['energy_MAE_eV_atom'] for v in report['summary'].values())
                    score = max(sulfur/.1, all_force/.03, energy/.025 if config['joint'] else 0.)
                    verdict = dict(rank=[False, score], progress_score=score,
                        passed=sulfur < .1 and all_force < .03 and (not config['joint'] or energy < .025),
                        sulfur_MAE=sulfur, all_force_MAE=all_force, energy_MAE=energy)
                else:
                    report, _ = evaluate(model, data, validation_ids)
                    verdict = acceptance(report['summary'], baseline['summary'], CONFIG['acceptance'], expected)
                    scheduler.step(verdict['progress_score'])
                row.update(validation=report['summary'], acceptance=verdict)
                old_score = state['best_score']
                if old_score is None or verdict['progress_score'] < old_score*(1-settings['meaningful_improvement']):
                    state['best_score'] = verdict['progress_score']
                    state['last_improved_epoch'] = completed_epoch
            state['epoch'] = completed_epoch
            state['sample'] = None
            state['cursor'] = 0
            state['history'].append(row)
            save(output / 'history.json', state['history'])
            if verdict is not None and (state['best_rank'] is None or verdict['rank'] < state['best_rank']):
                state['best_rank'] = verdict['rank']
                save_potential(model, output / 'model')
                save(output / 'validation.json', report)
                save(output / 'best.json', dict(epoch=completed_epoch, configuration=config, **verdict))
                atomic_checkpoint(output / 'best.pt', checkpoint())
            if not diagnostic or do_validation:
                persist('running')
            print(json.dumps(dict(run=config['name'], epoch=completed_epoch,
                acceptance=verdict, seconds=time.time()-epoch_start)), flush=True)
            if diagnostic and verdict and verdict['passed']:
                persist('completed')
                return
            if not diagnostic and completed_epoch-state['last_improved_epoch'] >= settings['stop_patience_epochs']:
                persist('early_stopped')
                return
        persist('completed')
    verify_model()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['diagnostic', 'screen', 'refine'])
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    index = args.index if args.index is not None else int(os.environ['SLURM_PROCID'])
    config = json.loads((HERE / 'experiments.json').read_text())[args.stage][index]
    try:
        train(config, float(os.environ['CAMPAIGN_DEADLINE']),
              HERE / 'runs' / config['name'] / 'last.pt' if args.resume else None)
    except BaseException as exc:
        save(HERE / 'runs' / config['name'] / 'failure.json',
             dict(error=str(exc), traceback=traceback.format_exc(), timestamp=time.time()))
        raise

if __name__ == '__main__':
    main()
