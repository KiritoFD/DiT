"""Single-arm U100 pipeline on the 4090: generic 100k pretrain -> Top10 FT.

Phases (resumable via status.json): rasters -> bank -> prepare -> configs ->
benchmark (memory gate) -> pretrain 100k -> prepare_ft -> ft -> eval_offline ->
summary. The pretraining is the pure generic pool (Top10 writer rows 0-9 stay
at their SupCon initialization); the FT makes the whole table trainable with a
20% generic replay and rows 0-9 receiving their Top10 data. Checkpoint/eval
every 5000 steps in both stages.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FT_STEPS = 25000


def pretrain_config(out, pool, manifest, steps):
    return dict(
        architecture='latent_cnn_unet_v1', objective='flow',
        channels=[192, 384, 576], num_res_blocks=4, decoder_res_blocks=None,
        skip_gate_scales=[], std_gate_scales=[], writer_film_rank=0, writer_film_layers=[],
        writer_film_spec=None, decouple_writer_condition=False, decoder_std_residual=False,
        decoder_std_scale=0.2, residual_cond_gate=False, stage_specific_cond=False,
        stage_cond_mode='adapter', act_fn='silu', requires_warm_start=False,
        num_writers=manifest['num_writers'], freeze_writer_table=False, writer_mode='shared',
        drop_writer=0.25,
        batch=256, workers=8, memory_format='channels_last',
        condition_tables_root=str(out / 'tables'),
        max_steps=steps, stop_at_step=steps, lr_decay_steps=steps, checkpoint_steps=[],
        seed=90, lr=6e-5, weight_decay=0.02, warmup=1500, min_lr_ratio=0.1, grad_clip=1.0,
        ema_decay=0.999, ema_interval=2, ema_warmup=True, ema_start_step=0,
        log_every=50, eval_every=5000, ckpt_every=5000,
        output=str(out / 'runs_100k'),
        data_csv=str(out / 'train_100k.csv'),
        latent_shards_dir=str(pool / 'latents'), img_root='',
        std_soft_root=str(pool / 'std_bbox64'),
        std_soft_sigma_min=0.0, std_soft_sigma_max=0.0, std_soft_eval_sigma=0.65,
        std_shape_cache=str(out / 'std_shape_bank.pt'), std_shape_aug='strong',
        char_remap_json=str(out / 'tables' / 'char_remap.json'),
        callig_remap_json=str(out / 'tables' / 'callig_remap.json'),
        font_remap_json=str(out / 'tables' / 'font_remap.json'),
        eval_csv=str(out / 'validation_val100.csv'),
        vae_path='pretrained_models/sd-vae-ft-ema', vae_scaling_factor=0.18215, vae_shift_factor=0.0,
        eval_batch=16, eval_steps=50, compile=True)


def ft_config(base, out, final_checkpoint):
    config = dict(base)
    config.update(
        data_csv=str(out / 'ft_train.csv'), eval_csv=str(out / 'validation_val100.csv'),
        std_soft_eval_sigma=0.0, output=str(out / 'ft_runs'),
        initialize_from=str(final_checkpoint), initialize_state='ema',
        freeze_writer_table=False,
        max_steps=FT_STEPS, stop_at_step=FT_STEPS, lr_decay_steps=FT_STEPS,
        lr=1.5e-5, warmup=300, ckpt_every=5000, eval_every=5000,
        eval_before_training=True, ema_start_step=0)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='assets/u100_pretrain_20261024')
    parser.add_argument('--pool', default='assets/generic48_kxl_top10_20261022')
    parser.add_argument('--reference', default='assets/reference_kxl_20261023/train.csv')
    parser.add_argument('--steps', type=int, default=75000)
    parser.add_argument('--memory-gate-gib', type=float, default=23.2)
    args = parser.parse_args()
    out = Path(args.out); pool = Path(args.pool)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    status_path = out / 'status.json'
    phases = ('rasters', 'bank', 'prepare', 'configs', 'benchmark',
              'pretrain_100k', 'prepare_ft', 'ft', 'eval_offline', 'summary')
    status = dict(pid=os.getpid(), started=time.strftime('%F %T'), phases={})
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status['pid'] = os.getpid()
        for stale in [k for k in status['phases'] if k not in phases]:
            del status['phases'][stale]
    done = status['phases']

    def save():
        status_path.write_text(json.dumps(status, indent=2))

    def run(name, cmd):
        done[name] = dict(state='running', started=time.strftime('%F %T'))
        save()
        with (out / 'logs' / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            done[name].update(state='failed', returncode=result.returncode)
            save(); raise SystemExit(f'phase {name} failed, see logs/{name}.log')
        done[name].update(state='done', finished=time.strftime('%F %T'))
        save()

    py = sys.executable
    save()
    if not done.get('rasters', {}).get('state') == 'done':
        run('rasters', [py, '-u', '-m', 'tools.u100_pretrain.build_rasters',
                        '--store', str(pool / 'std_bbox64'), '--out', str(out / 'rasters256.pt')])
    if not done.get('bank', {}).get('state') == 'done':
        run('bank', [py, '-u', '-m', 'tools.u100_pretrain.precompute_bank',
                     '--raw', str(out / 'rasters256.pt'), '--store', str(pool / 'std_bbox64'),
                     '--out', str(out / 'std_shape_bank.pt'), '--variants', '8'])
    if not done.get('prepare', {}).get('state') == 'done':
        run('prepare', [py, '-u', '-m', 'tools.u100_pretrain.prepare',
                        '--pool', str(pool), '--reference', str(args.reference), '--out', str(out)])
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    config = pretrain_config(out, pool, manifest, args.steps)
    (out / 'pretrain_100k.json').write_text(json.dumps(config, indent=2))
    if not done.get('benchmark', {}).get('state') == 'done':
        run('benchmark', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain_100k.json'),
                          '--benchmark', '20'])
        line = next(l for l in (out / 'logs' / 'benchmark.log').read_text().splitlines()
                    if l.startswith('BENCHMARK '))
        report = json.loads(line[len('BENCHMARK '):])
        if report['peak_allocated_GiB'] > args.memory_gate_gib:
            done['benchmark'].update(state='failed', report=report); save()
            raise SystemExit(f"memory gate: {report['peak_allocated_GiB']:.2f} GiB > {args.memory_gate_gib}")
        done['benchmark']['report'] = report; save()
    if not done.get('pretrain_100k', {}).get('state') == 'done':
        run('pretrain_100k', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'pretrain_100k.json')])
    final = sorted((out / 'runs_100k').glob('*/checkpoints/*.pt'))[-1]
    if not done.get('prepare_ft', {}).get('state') == 'done':
        run('prepare_ft', [py, '-u', '-m', 'tools.u100_pretrain.prepare_ft',
                           '--pool', str(pool), '--out', str(out)])
    (out / 'ft.json').write_text(json.dumps(ft_config(config, out, final), indent=2))
    if not done.get('ft', {}).get('state') == 'done':
        run('ft', [py, '-u', '-m', 'src.unet.train', '--config', str(out / 'ft.json')])
    ft_final = sorted((out / 'ft_runs').glob('*/checkpoints/*.pt'))[-1]
    if not done.get('eval_offline', {}).get('state') == 'done':
        run('eval_offline', [py, '-u', '-m', 'tools.u100_pretrain.eval_offline',
                             '--checkpoint', str(ft_final), '--out', str(out / 'eval_offline_ft'),
                             '--val100', str(out / 'validation_val100.csv')])
    summary = dict(pretrain=[json.loads(p.read_text()) for p in
                             sorted((out / 'runs_100k').glob('*/checkpoints/eval_auto_*.json'))],
                   finetune=[json.loads(p.read_text()) for p in
                             sorted((out / 'ft_runs').glob('*/checkpoints/eval_auto_*.json'))])
    (out / 'pipeline_summary.json').write_text(json.dumps(summary, indent=2))
    status['finished'] = time.strftime('%F %T'); save()
    print('U100_PIPELINE_DONE', out)


if __name__ == '__main__':
    main()
