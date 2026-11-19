"""Campaign-local paths and immutable provenance; never loads a remote Codex."""
from pathlib import Path
import contextlib, fcntl, hashlib, json, os, time

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'config.json').read_text())
MODEL = HERE.parents[1] / 'models/fine_tuned_m3gnet_model'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

@contextlib.contextmanager
def lock(name):
    path = HERE / 'locks' / (name + '.lock')
    path.parent.mkdir(exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def verify_model():
    actual = {n: sha(MODEL / n) for n in CONFIG['baseline_hashes']}
    if actual != CONFIG['baseline_hashes']:
        raise RuntimeError('Preserved baseline checkpoint changed')
    return actual

def dependencies():
    names = ['config.json', 'common.py', 'policy.py', 'objective.py', 'dataset.py',
             'metrics.py', 'train.py', 'runtime_adapter.py', 'calculator.py', 'dataset/manifest.json',
             'dataset/labels.traj']
    return {name: sha(HERE / name) for name in names}

def require_gate(name):
    result = json.loads((HERE / 'gates' / (name + '.json')).read_text())
    if not result.get('passed'):
        raise RuntimeError('Gate not passed: ' + name)
    for name, digest in result.get('dependencies', {}).items():
        if sha(HERE / name) != digest:
            raise RuntimeError('Stale gate dependency: ' + name)
    return result

def record_gate(name, **fields):
    result = dict(passed=True, timestamp=time.time(), dependencies=dependencies(), **fields)
    save(HERE / 'gates' / (name + '.json'), result)
    return result
