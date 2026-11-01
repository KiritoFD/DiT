"""Inventory every generated panel whose PNGs still exist, and emit an eval_unpaired
jobs file so E-G can be recomputed with the official implementation.

Only panels with a complete `g0..g<n-1>.png` set under
`eval_samples_ctrl/step*/eval200fix` qualify. The panel csv is resolved by path
convention and then verified against the recorded `csv_sha256` where one exists, so a
silently swapped panel cannot slip through.

  python tools/poster_20261101/build_eg_jobs.py --root . --out jobs.json
"""
import argparse
import hashlib
import json
from pathlib import Path

VAL100 = 'assets/u100_pretrain_20261024/validation_val100.csv'
N187 = 'assets/style_focus_85k_20261029/4090/protocols/S_187_seed{n}.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_for(panel_dir):
    """Panel csv by convention: heldout187_seedN -> S_187_seedN, else val100."""
    name = panel_dir.name
    if 'heldout187_seed' in name:
        seed = name.rsplit('seed', 1)[-1]
        return N187.format(n=seed)
    return VAL100


def arm_of(panel_dir):
    """Human label: nearest 'runs/<ARM>' or 'eval/<ARM>' ancestor."""
    parts = panel_dir.parts
    for i, p in enumerate(parts):
        if p in ('runs', 'eval') and i + 1 < len(parts):
            return parts[i + 1]
    return panel_dir.parent.name


def step_of(panel_dir):
    for p in panel_dir.parts:
        if p.startswith('step') and p[4:].isdigit():
            return int(p[4:])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='assets/metrics_index/eg_jobs.json')
    ap.add_argument('--report', default='assets/metrics_index/eg_inventory.json')
    a = ap.parse_args()
    root = Path(a.root)

    jobs, skipped = [], []
    seen = set()
    for batch in root.rglob('eval_stdskel_batch.csv'):
        d = batch.parent
        ctrls = list((d / 'eval_samples_ctrl').glob('step*'))
        if not ctrls:
            continue
        step_dir = sorted(ctrls)[-1]
        panel = step_dir / 'eval200fix'
        if not panel.is_dir():
            continue
        try:
            import csv as _csv
            n = sum(1 for _ in _csv.DictReader(open(batch, encoding='utf-8-sig')))
        except Exception as e:
            skipped.append((str(d), f'batch unreadable: {e}'))
            continue
        if n == 0:
            skipped.append((str(d), 'empty batch'))
            continue
        missing = [i for i in range(n) if not (panel / f'g{i}.png').is_file()]
        if missing:
            skipped.append((str(d), f'{len(missing)}/{n} png missing'))
            continue
        csv_path = csv_for(d)
        if not Path(csv_path).is_file():
            skipped.append((str(d), f'panel csv not found: {csv_path}'))
            continue
        try:
            import csv as _csv
            n_csv = sum(1 for _ in _csv.DictReader(open(csv_path, encoding='utf-8-sig')))
        except Exception as e:
            skipped.append((str(d), f'csv unreadable: {e}'))
            continue
        if n_csv != n:
            skipped.append((str(d), f'csv rows {n_csv} != png {n}'))
            continue
        key = (arm_of(d), step_of(d), d.name)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(dict(arm=arm_of(d), step=step_of(d), protocol=d.name,
                         csv=csv_path, samples=str(panel), n=n,
                         csv_sha256=sha(csv_path), path=str(d)))

    # verify the recorded csv_sha256 where the eval wrote one
    for j in jobs:
        req = Path(j['path']) / 'request.json'
        if req.is_file():
            try:
                rec = json.loads(req.read_text()).get('csv_sha256')
            except Exception:
                rec = None
            if rec and rec != j['csv_sha256']:
                j['csv_mismatch'] = True

    jobs.sort(key=lambda j: (j['arm'], j['step'], j['protocol']))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps([{k: v for k, v in j.items() if k != 'path'}
                                       for j in jobs], ensure_ascii=False, indent=1),
                           encoding='utf-8')
    Path(a.report).write_text(json.dumps(dict(n_jobs=len(jobs), jobs=jobs,
                                              skipped=skipped),
                                         ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(jobs)} panels with complete images -> {a.out}')
    for j in jobs:
        flag = ' CSV-MISMATCH' if j.get('csv_mismatch') else ''
        print(f"  {j['arm']:<24} n={j['n']:<4} {j['protocol']}{flag}")
    print(f'{len(skipped)} skipped -> {a.report}')
    for p, why in skipped[:15]:
        print(f'  skip {p}: {why}')


if __name__ == '__main__':
    main()
