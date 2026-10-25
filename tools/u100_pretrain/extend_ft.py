"""Extend the Top10 fine-tune (default 25k -> 50k) with full state resume.

Raw/EMA/Adam/scheduler/RNG all restore from the source checkpoint; the cosine
horizon is explicitly lengthened, so LR rejoins mid-schedule (an audited
schedule change, not an uninterrupted original trajectory). After training the
offline three-protocol evaluation runs on the final checkpoint.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='assets/u100_pretrain_20261024')
    parser.add_argument('--from-step', type=int, default=25000)
    parser.add_argument('--total', type=int, default=50000)
    args = parser.parse_args()
    bundle = Path(args.bundle)
    src_run = sorted((bundle / 'ft_runs').glob('*-Unet-flow'))[-1]
    resume = src_run / 'checkpoints' / f'{args.from_step:07d}.pt'
    if not resume.exists(): raise FileNotFoundError(resume)

    config = json.loads((bundle / 'ft.json').read_text(encoding='utf-8'))
    old_horizon = config['max_steps']
    if args.total <= old_horizon: raise ValueError('Extension must increase the horizon')
    config.update(max_steps=args.total, stop_at_step=args.total, lr_decay_steps=args.total,
                  output=str(bundle / 'ft_ext_runs'), initialize_from='',
                  eval_before_training=False)
    (bundle / 'ft_ext.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    status = dict(phase='training', pid=os.getpid(),
                  resume=str(resume), old_horizon=old_horizon, new_horizon=args.total,
                  lr_policy='Rejoin cosine at the extended horizon; explicit schedule change')

    def save():
        (bundle / 'ft_ext_status.json').write_text(json.dumps(status, indent=2))

    save()
    with (bundle / 'logs' / 'ft_ext.log').open('w') as log:
        result = subprocess.run([sys.executable, '-u', '-m', 'src.unet.train',
                                 '--config', str(bundle / 'ft_ext.json'), '--resume', str(resume)],
                                cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        status['phase'] = 'failed'; save()
        raise SystemExit('extension training failed, see logs/ft_ext.log')

    final = sorted((bundle / 'ft_ext_runs').glob('*/checkpoints/*.pt'))[-1]
    status['phase'] = 'eval'; status['final'] = str(final); save()
    with (bundle / 'logs' / 'ft_ext_eval.log').open('w') as log:
        result = subprocess.run([sys.executable, '-u', '-m', 'tools.u100_pretrain.eval_offline',
                                 '--checkpoint', str(final), '--out', str(bundle / 'eval_offline_ft_ext'),
                                 '--val100', str(bundle / 'validation_val100.csv')],
                                cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
    status['phase'] = 'completed' if result.returncode == 0 else 'eval_failed'
    status['finished'] = time.strftime('%F %T'); save()


if __name__ == '__main__':
    main()
