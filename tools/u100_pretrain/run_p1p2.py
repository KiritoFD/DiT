"""P1 (writer-aware mixed) + P2 (Top10 specialist with pair sampling) on the
reused 114M generic pretraining checkpoint.

P0 = assets/u100_pretrain_20261024/runs_100k 0075000 EMA (kxl-100k pool;
no 400k checkpoint exists — substitution recorded).
P1 = 10k: 70% generic rows with NULL writer row + 30% Top10 real rows.
P2 = 20k: Top10 only, 70% ordinary + 30% same-char same-script cross-writer
     pairs (PairMixtureSampler, plain FM loss on every sample).
Backbone LR 1e-5, writer/condition LR 5e-5; B512; strong STD aug kept for
continuity with the P0 checkpoint's input protocol.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}
BUNDLE = 'assets/u100_pretrain_20261024'


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='assets/u100_p1p2_20261025')
    parser.add_argument('--p1-steps', type=int, default=10000)
    parser.add_argument('--p2-steps', type=int, default=20000)
    parser.add_argument('--memory-gate-gib', type=float, default=23.2)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    status_path = out / 'status.json'
    status = dict(pid=os.getpid(), started=time.strftime('%F %T'), phases={})
    if status_path.exists():
        status = json.loads(status_path.read_text()); status['pid'] = os.getpid()
    done = status['phases']

    def save(): status_path.write_text(json.dumps(status, indent=2))

    def run(name, cmd):
        done[name] = dict(state='running', started=time.strftime('%F %T')); save()
        with (out / 'logs' / f'{name}.log').open('w') as log:
            r = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            done[name].update(state='failed', returncode=r.returncode); save()
            raise SystemExit(f'phase {name} failed')
        done[name].update(state='done', finished=time.strftime('%F %T')); save()

    py = sys.executable
    save()

    if not done.get('prepare', {}).get('state') == 'done':
        generic = read(f'{BUNDLE}/train_100k.csv')
        fn = list(generic[0].keys())
        rng = np.random.default_rng(90)
        # P1: generic -> null row 259; Top10 duplicated to 30% share
        g1 = [dict(r, calligrapher_id='259') for r in generic]
        t10 = read(f'{BUNDLE}/../generic48_kxl_top10_20261022/top10_train.csv')
        t10 = [r for r in t10 if r['generic'] == '0']
        # align to pool schema
        t10 = [{k: r.get(k, '') for k in fn} for r in t10]
        t10 = [dict(r, calligrapher_id=str(TOP10_RAW[int(r['calligrapher_id'])])) for r in t10]
        n_dup = round(len(g1) * 3 / 7)
        idx = rng.choice(len(t10), n_dup, replace=True)
        p1_rows = g1 + [t10[i] for i in idx]
        write(out / 'p1_train.csv', p1_rows, fn)
        # P1 remap must contain the explicit null row 259
        tp = out / 'tables_p1'; tp.mkdir(exist_ok=True)
        identity = {str(g): g for g in range(260)}
        (tp / 'callig_remap.json').write_text(json.dumps(identity))
        for name in ('char_remap.json', 'font_remap.json', 'font_table.npy', 'callig_table.npy'):
            (tp / name).write_bytes((Path(BUNDLE) / 'tables' / name).read_bytes())
        # P2: Top10 clean only (fields incl. calligrapher name for pair sampler)
        write(out / 'p2_train.csv', t10, fn)
        (out / 'data_manifest.json').write_text(json.dumps(dict(
            p1_rows=len(p1_rows), p1_generic=len(g1), p1_top10_dupes=int(n_dup),
            p2_rows=len(t10), null_row=259,
            p0=f'{BUNDLE}/runs_100k 0075000 EMA (kxl-100k substitution; no 400k ckpt exists)'), indent=1))
        done['prepare'] = dict(state='done'); save()
    manifest = json.loads((out / 'data_manifest.json').read_text())

    base = json.loads(Path(f'{BUNDLE}/pretrain_100k.json').read_text(encoding='utf-8'))
    p0_ckpt = sorted(Path(f'{BUNDLE}/runs_100k').glob('*/checkpoints/0075000.pt'))[-1]

    p1 = dict(base)
    p1.update(max_steps=args.p1_steps, stop_at_step=args.p1_steps, lr_decay_steps=args.p1_steps,
              output=str(out / 'p1_runs'), data_csv=str(out / 'p1_train.csv'),
              callig_remap_json=str(out / 'tables_p1' / 'callig_remap.json'),
              char_remap_json=str(out / 'tables_p1' / 'char_remap.json'),
              font_remap_json=str(out / 'tables_p1' / 'font_remap.json'),
              condition_tables_root=str(out / 'tables_p1'),
              initialize_from=str(p0_ckpt), initialize_state='ema',
              lr=1e-5, writer_cond_lr=5e-5, warmup=200, null_writer_rows=True,
              drop_writer=0.25, batch=512, ckpt_every=5000, eval_every=5000, eval_before_training=False)
    (out / 'p1.json').write_text(json.dumps(p1, indent=2))

    p2 = dict(p1)
    p2.update(max_steps=args.p2_steps, stop_at_step=args.p2_steps, lr_decay_steps=args.p2_steps,
              output=str(out / 'p2_runs'), data_csv=str(out / 'p2_train.csv'),
              drop_writer=0.10, pair_fraction=0.3, null_writer_rows=False,
              callig_remap_json=f'{BUNDLE}/tables/callig_remap.json',
              char_remap_json=f'{BUNDLE}/tables/char_remap.json',
              font_remap_json=f'{BUNDLE}/tables/font_remap.json',
              condition_tables_root=f'{BUNDLE}/tables',
              ckpt_every=5000, eval_every=5000, eval_before_training=False)

    if not done.get('benchmark', {}).get('state') == 'done':
        run('benchmark', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'p1.json'), '--benchmark', '20'])
        line = next(l for l in (out / 'logs' / 'benchmark.log').read_text().splitlines() if l.startswith('BENCHMARK '))
        report = json.loads(line[len('BENCHMARK '):])
        if report['peak_allocated_GiB'] > args.memory_gate_gib:
            raise SystemExit(f"memory gate {report['peak_allocated_GiB']:.2f}")
        done['benchmark']['report'] = report; save()
    if not done.get('p1', {}).get('state') == 'done':
        run('p1', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'p1.json')])
    p1_final = sorted((out / 'p1_runs').glob('*/checkpoints/*.pt'))[-1]
    p2.update(initialize_from=str(p1_final))
    (out / 'p2.json').write_text(json.dumps(p2, indent=2))
    if not done.get('p2', {}).get('state') == 'done':
        run('p2', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'p2.json')])
    p2_final = sorted((out / 'p2_runs').glob('*/checkpoints/*.pt'))[-1]
    if not done.get('eval_offline', {}).get('state') == 'done':
        run('eval_offline', [py, '-u', '-m', 'tools.u100_pretrain.eval_offline',
                             '--checkpoint', str(p2_final), '--out', str(out / 'eval_offline_p2final'),
                             '--val100', f'{BUNDLE}/validation_val100.csv'])
    summary = dict(p1=[json.loads(p.read_text()) for p in sorted((out / 'p1_runs').glob('*/checkpoints/eval_auto_*.json'))],
                   p2=[json.loads(p.read_text()) for p in sorted((out / 'p2_runs').glob('*/checkpoints/eval_auto_*.json'))])
    (out / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2))
    status['finished'] = time.strftime('%F %T'); save()
    print('P1P2_DONE', out)


if __name__ == '__main__':
    main()
