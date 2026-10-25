"""48M control arm: today's pipeline verbatim on the established 48M U-Net.

Reuses the 114M bundle's data assets (tables, train/ft CSVs, STD bank,
validation); only the architecture (96/192/256 x8 blocks, 48.3M) and output
dirs differ. Same budgets: 75k pretrain + 25k FT, ckpt/eval every 5k,
drop_writer .25. Serves as the matched-capacity baseline for the continuous
style-conditioned arm.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHANNELS = [96, 192, 256]
BLOCKS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='assets/u100_pretrain_20261024')
    parser.add_argument('--out', default='assets/u100_control48_20261024')
    parser.add_argument('--steps', type=int, default=75000)
    parser.add_argument('--ft-steps', type=int, default=25000)
    args = parser.parse_args()
    bundle = Path(args.bundle); out = Path(args.out)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
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
    pre = json.loads((bundle / 'pretrain_100k.json').read_text(encoding='utf-8'))
    pre.update(channels=CHANNELS, num_res_blocks=BLOCKS, max_steps=args.steps,
               stop_at_step=args.steps, lr_decay_steps=args.steps, output=str(out / 'runs'))
    (out / 'pretrain48.json').write_text(json.dumps(pre, indent=2))
    if not done.get('benchmark', {}).get('state') == 'done':
        run('benchmark', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain48.json'), '--benchmark', '20'])
        line = next(l for l in (out / 'logs' / 'benchmark.log').read_text().splitlines() if l.startswith('BENCHMARK '))
        report = json.loads(line[len('BENCHMARK '):])
        if report['peak_allocated_GiB'] > 23.2:
            raise SystemExit('memory gate')
        done['benchmark']['report'] = report; save()
    if not done.get('pretrain', {}).get('state') == 'done':
        run('pretrain', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain48.json')])
    final = sorted((out / 'runs').glob('*/checkpoints/*.pt'))[-1]
    ft = json.loads((bundle / 'ft.json').read_text(encoding='utf-8'))
    ft.update(channels=CHANNELS, num_res_blocks=BLOCKS, max_steps=args.ft_steps,
              stop_at_step=args.ft_steps, lr_decay_steps=args.ft_steps,
              initialize_from=str(final), initialize_state='ema', output=str(out / 'ft_runs'))
    (out / 'ft48.json').write_text(json.dumps(ft, indent=2))
    if not done.get('ft', {}).get('state') == 'done':
        run('ft', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'ft48.json')])
    ft_final = sorted((out / 'ft_runs').glob('*/checkpoints/*.pt'))[-1]
    if not done.get('eval_offline', {}).get('state') == 'done':
        run('eval_offline', [py, '-u', '-m', 'tools.u100_pretrain.eval_offline',
                             '--checkpoint', str(ft_final), '--out', str(out / 'eval_offline_ft'),
                             '--val100', str(bundle / 'validation_val100.csv')])
    summary = dict(pretrain=[json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*/checkpoints/eval_auto_*.json'))],
                   finetune=[json.loads(p.read_text()) for p in sorted((out / 'ft_runs').glob('*/checkpoints/eval_auto_*.json'))])
    (out / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2))
    status['finished'] = time.strftime('%F %T'); save()
    print('CONTROL48_DONE', out)


if __name__ == '__main__':
    main()
