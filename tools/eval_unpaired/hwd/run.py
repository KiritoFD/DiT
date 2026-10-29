"""CPU-only official-weight HWD on existing, identity-checked generation panels."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from tools.eval_unpaired.protocol import read, save, csv_rows
from .model import load, extract, distance, UPSTREAM, URL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', nargs='+', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    weights = Path(a.weights)
    contract = dict(upstream=UPSTREAM, url=URL, weights_size=weights.stat().st_size,
                    preprocessing='RGB, white square pad, nearest height32, ToTensor[0,1]; batch1',
                    distance='macro writer L2 of mean unnormalized VGG features', device='cpu')
    if (out/'contract.json').exists() and read(out/'contract.json') != contract:
        raise ValueError('HWD contract changed')
    net = load(weights)  # Strictly validates the complete official 10400-class checkpoint.
    save(out/'contract.json', contract)
    results = []; seen = set()
    for value in a.roots:
        root = Path(value)
        for old in read(root/'summary.json'):
            job = old['job']; key = (job['arm'], job['step'], job['protocol'])
            if key in seen: continue
            seen.add(key)
            rows = csv_rows(job['csv'])
            if hashlib.sha256(Path(job['csv']).read_bytes()).hexdigest() != job['csv_sha256']:
                raise ValueError('CSV identity changed')
            if len(rows) != job['n']: raise ValueError('Sample count mismatch')
            name = '_'.join(map(str, key)); dest = out/name; dest.mkdir(exist_ok=True)
            files = [Path(job['samples'])/f'{prefix}{i}.png'
                     for prefix in ['g', 'gt'] for i in range(len(rows))]
            inputs = dict(job=job, images=[dict(path=str(f), size=f.stat().st_size,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files])
            if (dest/'inputs.json').exists() and read(dest/'inputs.json') != inputs:
                raise ValueError('HWD image identity changed')
            save(dest/'inputs.json', inputs)
            if (dest/'result.json').exists():
                results.append(read(dest/'result.json')); continue
            features = []
            for prefix in ['g', 'gt']:
                cache = dest/f'{prefix}.npy'
                if not cache.exists():
                    np.save(cache, extract(net, [Path(job['samples'])/f'{prefix}{i}.png'
                                               for i in range(len(rows))]))
                features.append(np.load(cache))
            result = dict(job=job, paired_panel=distance(*features, rows),
                interpretation='Official HWD mechanism on square Chinese tiles; paired GT composition, not unseen-source or aesthetic proof')
            save(dest/'result.json', result); results.append(result)
            save(out/'summary.json', results)
            print('HWD_COMPLETE', name, result['paired_panel']['hwd'], flush=True)
    save(out/'summary.json', results); save(out/'complete.json', dict(jobs=len(results)))


if __name__ == '__main__': main()
