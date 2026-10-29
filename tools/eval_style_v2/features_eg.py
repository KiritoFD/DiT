"""E-G feature access for eval_style_v2: reuse the frozen ResNet18 evaluator.

Prefers the eval_unpaired per-panel caches (identical values to the published
domain metrics); computes on CPU only when no cache exists. No IQA load.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import sha256_file,save_json

EG_CACHE_DIRS={'48':['assets/eval_unpaired_20261029/48_style','assets/eval_unpaired_20261029/48',
                     'assets/eval_unpaired_20261029/S11_v1'],
               '4090':['assets/eval_unpaired_20261029/4090_style','assets/eval_unpaired_20261029/4090'],
               'default':sorted(__import__('pathlib').Path('.').glob('assets/eval_unpaired_20261029/*'))}


def _bank_source(machine='48'):
    dirs=EG_CACHE_DIRS.get(machine) or EG_CACHE_DIRS['default']
    for d in dirs:
        root=Path(d)
        rows=root/'reference_rows.json';feats=root/'reference_features.npz'
        if rows.is_file() and feats.is_file():
            import numpy as np
            z=np.load(feats)
            if 'E-G' in z:
                return root,json.loads(rows.read_text()),z['E-G']
    raise FileNotFoundError('No reference_rows.json + E-G features under '+', '.join(EG_CACHE_DIRS))


def bank_eg(machine='48'):
    """(rows, features) of the frozen reference bank, byte-identical to 1029 metrics."""
    root,rows,feats=_bank_source(machine)
    assert len(rows)==len(feats),(len(rows),feats.shape)
    return rows,np.asarray(feats,dtype=np.float32)


def _torch_threads(n=4):
    import torch
    torch.set_num_threads(n)


def eg_features(paths,cache_path=None):
    """E-G features for arbitrary paths; npz cache keyed by the path list sha."""
    import torch
    key=hashlib.sha256('\n'.join(sorted(paths)).encode()).hexdigest()[:16]
    if cache_path is None:
        from .common import cache_dir
        cache_path=cache_dir('eg')/f'eg_{key}.npz'
    cache_path=Path(cache_path)
    if cache_path.is_file():
        return np.load(cache_path)['E-G']
    _torch_threads(4)
    from tools.eval_unpaired.networks import Evaluators
    nets=Evaluators(device='cpu',iqa=False)
    out=nets.extract(paths,32)
    feats=np.asarray(out['E-G'],dtype=np.float32)
    if len(feats)!=len(paths):raise ValueError('feature count mismatch')
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(cache_path,**{'E-G':feats})
    return feats


def eg_provenance(machine='48'):
    try:
        root,rows,_=_bank_source(machine)
        return dict(bank_root=str(root),n_bank=len(rows))
    except FileNotFoundError as e:
        return dict(error=str(e))
