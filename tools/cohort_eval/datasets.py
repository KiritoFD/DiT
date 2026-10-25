"""Build nested, writer-balanced training probes without changing frozen splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

DEFAULT_ROOT = Path('assets/data_cohorts/top20_20261021_v1')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        items = list(csv.DictReader(f))
    if not items:
        raise ValueError(f'Empty collection: {path}')
    return items


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_key(row):
    return row['image_path'].strip().replace('\\', '/')


def seeded_rows(items, seed=90):
    """Preserve frozen seeds; fill missing seeds by image identity, never row index."""
    result = []
    for row in items:
        row = dict(row)
        if not str(row.get('eval_noise_seed', '')).strip():
            token = f'{seed}|{image_key(row)}'.encode()
            row['eval_noise_seed'] = str(int.from_bytes(hashlib.sha256(token).digest()[:8], 'big') % 2**63)
        value = int(row['eval_noise_seed'])
        if not 0 <= value < 2**63:
            raise ValueError('Invalid eval_noise_seed')
        result.append(row)
    return result


def write_rows(path, items):
    fields = list(dict.fromkeys(k for r in items for k in r))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(items)


def select_scripts(pool, reference, count, seed):
    """Closest feasible test-script quotas; deterministic sampling without replacement."""
    groups = defaultdict(list)
    for row in pool:
        groups[row['script_id']].append(row)
    if len({image_key(r) for r in pool}) != len(pool) or len(pool) < count:
        raise ValueError('Training pool has duplicates or insufficient images')
    target = Counter(r['script_id'] for r in reference)
    desired = {s: count * n / len(reference) for s, n in target.items()}
    quotas = {s: 0 for s in groups}
    for _ in range(count):
        options = [s for s in groups if quotas[s] < len(groups[s])]
        chosen = max(options, key=lambda s: (desired.get(s, 0)-quotas[s], s))
        quotas[chosen] += 1
    chosen = []
    for script in sorted(groups):
        candidates = sorted(groups[script], key=lambda r: hashlib.sha256(
            f'{seed}|{image_key(r)}'.encode()).hexdigest())
        chosen.extend(candidates[:quotas[script]])
    return chosen


def build_probes(root=DEFAULT_ROOT, output=None, seed=90):
    root = Path(root)
    output = Path(output) if output else root/'train_probes_v1'
    if output.exists():
        raise FileExistsError(output)
    data = root/'datasets_final'
    train, test = read_rows(data/'top20/train.csv'), read_rows(data/'top20/test.csv')
    validation = read_rows(data/'top20/validation.csv')
    train_keys = {image_key(r) for r in train}
    assert not train_keys & {image_key(r) for r in test+validation}
    pools, refs = defaultdict(list), defaultdict(list)
    for row in train: pools[row['calligrapher_id']].append(row)
    for row in test: refs[row['calligrapher_id']].append(row)
    equal, half = [], []
    for writer, reference in refs.items():
        count = len(reference)
        if count % 2:
            raise ValueError('Half-size writer-balanced probe requires even per-writer test size')
        selected = select_scripts(pools[writer], reference, count, seed)
        equal.extend(selected)
        half.extend(select_scripts(selected, reference, count//2, seed))
    equal, half = seeded_rows(equal, seed), seeded_rows(half, seed)
    output.mkdir(parents=True)
    manifest = dict(seed=seed, role='training-fit diagnostics, not validation/test',
                    frozen_data_manifest_sha256=sha(data/'manifest.json'), levels={})
    previous = {name:set() for name in ('train_equal','train_half')}
    for size in (2,5,10,20):
        folder = data/f'top{size}'
        source, target = read_rows(folder/'train.csv'), read_rows(folder/'test.csv')
        writers = {r['calligrapher_id'] for r in target}
        allowed = {image_key(r) for r in source}
        manifest['levels'][str(size)] = dict(train_sha256=sha(folder/'train.csv'),
            test_sha256=sha(folder/'test.csv'), files={})
        for name, selected, divisor in (('train_equal',equal,1),('train_half',half,2)):
            items = [r for r in selected if r['calligrapher_id'] in writers]
            keys = {image_key(r) for r in items}
            assert len(keys) == len(items) == len(target)//divisor
            assert keys <= allowed and previous[name] <= keys
            previous[name] = keys
            path = output/f'top{size}_{name}.csv'
            write_rows(path, items)
            manifest['levels'][str(size)]['files'][name] = dict(path=path.name,
                n=len(items),sha256=sha(path),writer_script_counts=dict(Counter(
                    r['calligrapher_id']+':'+r['script_id'] for r in items)))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def resolve_collection(spec, root=DEFAULT_ROOT, probes=None):
    root = Path(root)
    if Path(spec).is_file():
        return Path(spec)
    if '/' in spec and spec.split('/')[0] in ('top2','top5','top10','top20'):
        level, name = spec.split('/')
        if name in ('test','validation'):
            return root/'datasets_final'/level/f'{name}.csv'
        if name in ('train_equal','train_half'):
            return (Path(probes) if probes else root/'train_probes_v1')/f'{level}_{name}.csv'
        raise ValueError(f'Unknown collection: {spec}')
    return Path(spec)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--output')
    p.add_argument('--seed',type=int,default=90)
    a=p.parse_args();manifest=build_probes(a.root,a.output,a.seed)
    print(json.dumps({k:{n:v['n'] for n,v in level['files'].items()}
                      for k,level in manifest['levels'].items()},indent=2))


if __name__=='__main__':main()
