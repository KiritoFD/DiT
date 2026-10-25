"""Arm B: 48M writer_slot style-conditioned pretraining, queued after arm A'.

Waits for the control48 pipeline to finish, then: merge the Top10 bank
(writer-order remap) -> calibrate the SupCon weight from measured gradient
ratios -> benchmark -> 75k pretrain -> 25k Top10 fine-tune (same data mix and
budgets as arm A') -> offline three-protocol eval. Everything except the
conditioning route (continuous cross-char style via the FiLM reader pathway
vs discrete writer ids) is identical between the arms.
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
    parser.add_argument('--control', default='assets/u100_control48_20261024')
    parser.add_argument('--out', default='assets/u100_style48_20261024')
    parser.add_argument('--steps', type=int, default=75000)
    parser.add_argument('--ft-steps', type=int, default=25000)
    args = parser.parse_args()
    bundle = Path(args.bundle); control = Path(args.control); out = Path(args.out)
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
            raise SystemExit(f'phase {name} failed, see logs/{name}.log')
        done[name].update(state='done', finished=time.strftime('%F %T')); save()

    py = sys.executable
    save()

    if not done.get('wait_control', {}).get('state') == 'done':
        done['wait_control'] = dict(state='running', started=time.strftime('%F %T')); save()
        while True:
            s = json.loads((control / 'status.json').read_text())
            if s.get('finished') or s['phases'].get('eval_offline', {}).get('state') == 'done':
                break
            time.sleep(300)
        done['wait_control'].update(state='done', finished=time.strftime('%F %T')); save()

    if not done.get('merge', {}).get('state') == 'done':
        run('merge', [py, '-u', '-m', 'tools.u100_pretrain.merge_top10_bank',
                      '--out', str(bundle / 'style_bank_top10.npz')])
    if not done.get('calibrate', {}).get('state') == 'done':
        run('calibrate', [py, '-u', '-m', 'tools.u100_pretrain.calibrate_lambda',
                          '--out', str(bundle / 'lambda_calibration.json')])
    lam = json.loads((bundle / 'lambda_calibration.json').read_text())['recommended_lambda']

    base = json.loads((bundle / 'pretrain_100k.json').read_text(encoding='utf-8'))
    base.update(channels=CHANNELS, num_res_blocks=BLOCKS, max_steps=args.steps,
                stop_at_step=args.steps, lr_decay_steps=args.steps, output=str(out / 'runs'),
                writer_mode='none', decouple_writer_condition=False, freeze_writer_table=True,
                drop_writer=.08, reference_dim=128, reference_injection='writer_slot',
                reference_dropout=.25, reference_runtime='group',
                reference_cache=str(bundle / 'style_bank.npz'),
                reference_extra_cache=str(bundle / 'style_bank_top10.npz'),
                reference_max_refs=2, reference_con_weight=lam,
                reference_temperature=.1, reference_hard_weight=2.)
    (out / 'pretrain.json').write_text(json.dumps(base, indent=2))
    if not done.get('benchmark', {}).get('state') == 'done':
        run('benchmark', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain.json'), '--benchmark', '20'])
        line = next(l for l in (out / 'logs' / 'benchmark.log').read_text().splitlines() if l.startswith('BENCHMARK '))
        report = json.loads(line[len('BENCHMARK '):])
        if report['peak_allocated_GiB'] > 23.2: raise SystemExit('memory gate')
        done['benchmark']['report'] = report; save()
    if not done.get('pretrain', {}).get('state') == 'done':
        run('pretrain', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain.json')])
    final = sorted((out / 'runs').glob('*/checkpoints/*.pt'))[-1]

    ft = dict(base)
    ft.update(data_csv=str(bundle / 'ft_train.csv'), std_soft_eval_sigma=0.0,
              output=str(out / 'ft_runs'), initialize_from=str(final), initialize_state='ema',
              max_steps=args.ft_steps, stop_at_step=args.ft_steps, lr_decay_steps=args.ft_steps,
              lr=1.5e-5, warmup=300, eval_before_training=True, ema_start_step=0)
    (out / 'ft.json').write_text(json.dumps(ft, indent=2))
    if not done.get('ft', {}).get('state') == 'done':
        run('ft', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'ft.json')])
    ft_final = sorted((out / 'ft_runs').glob('*/checkpoints/*.pt'))[-1]
    if not done.get('eval_offline', {}).get('state') == 'done':
        run('eval_offline', [py, '-u', '-m', 'tools.u100_pretrain.eval_offline',
                             '--checkpoint', str(ft_final), '--out', str(out / 'eval_offline_ft'),
                             '--val100', str(bundle / 'validation_val100.csv')])
    summary = dict(pretrain=[json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*/checkpoints/eval_auto_*.json'))],
                   finetune=[json.loads(p.read_text()) for p in sorted((out / 'ft_runs').glob('*/checkpoints/eval_auto_*.json'))])
    (out / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2))
    status['finished'] = time.strftime('%F %T'); save()
    print('STYLE48_DONE', out)


if __name__ == '__main__':
    main()
