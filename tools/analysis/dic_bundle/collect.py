"""Collect existing evaluation artifacts only; never load a model or touch training."""
import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def collect(root, host, output):
    root = Path(root)
    registry = []
    written = set()
    common_written = False
    with ZipFile(output, 'w', ZIP_DEFLATED, compresslevel=3) as archive:
        def put(name, data):
            if name not in written:
                archive.writestr(name, data)
                written.add(name)

        def add(model_id, label, run, step, state='EMA', cfg=1.0, family='', training_run=None):
            nonlocal common_written
            run = root / run
            training = root / training_run if training_run else run
            folder = run / f'eval_samples_ctrl/step{step:07d}/eval200fix'
            if not folder.exists():
                print('MISSING', model_id, str(folder), flush=True)
                return
            files = sorted(p for p in folder.iterdir() if re.fullmatch(r'g\d+\.png', p.name))
            indices = sorted(int(p.stem[1:]) for p in files)
            if indices != list(range(187)):
                raise ValueError(f'{model_id}: expected 187 indexed original PNGs, got {indices}')
            base = f'evidence/{model_id}'
            for p in files:
                put(f'images/originals/{model_id}/{p.name}', p.read_bytes())
            config_file = training / 'resolved_config.json'
            config = json.loads(config_file.read_text()) if config_file.exists() else {}
            if config:
                put(f'{base}/config.json', config_file.read_bytes())
            for original, target in [('eval_stdskel_summary.csv', 'metrics.csv'),
                                     ('eval_stdskel_batch.csv', 'per_image.csv')]:
                p = run / original
                if p.exists():
                    put(f'{base}/{target}', p.read_bytes())
            protocol = run / f'eval_protocol/step{step:07d}_eval200fix.json'
            protocol_data = json.loads(protocol.read_text()) if protocol.exists() else {}
            if protocol_data:
                put(f'{base}/protocol.json', protocol.read_bytes())
            summary_file = run / 'eval_stdskel_summary.csv'
            metrics = None
            if summary_file.exists():
                for row in csv.DictReader(summary_file.open(encoding='utf-8-sig')):
                    if int(row['step']) == step and row.get('set') == 'eval200fix':
                        metrics = row
            log = training / 'train.log'
            lines = log.read_text(errors='replace').splitlines() if log.exists() else []
            source = training / 'source_manifest.json'
            source_data = json.loads(source.read_text()) if source.exists() else {}
            resume = training / 'resume_record.json'
            identity_files = list((training / 'records').glob('*.identity.json'))
            identity = json.loads(identity_files[0].read_text()) if identity_files else {}
            # Offline evaluations sometimes write the selected metadata only into checkpoint args.
            # Do not fabricate config values if the parent config was not retained.
            item = dict(id=model_id, label=label, family=family, host=host, run=str(run),
                        training_run=str(training), step=step, state=state, cfg=cfg,
                        n=len(files), config=config, metrics=metrics, protocol=protocol_data)
            provenance = dict(collected_utc=datetime.now(timezone.utc).isoformat(),
                              source=item, log_start=lines[:12], log_end=lines[-12:],
                              source_manifest=source_data, identity=identity,
                              resume_record=json.loads(resume.read_text()) if resume.exists() else None,
                              retained_checkpoint_names=[p.name for p in (training / 'checkpoints').glob('*.pt')])
            put(f'{base}/provenance.json', json.dumps(provenance, ensure_ascii=False, indent=2).encode())
            if not common_written and host == '48':
                gt_files = [folder / f'gt{i}.png' for i in range(187)]
                if all(p.exists() for p in gt_files):
                    for i, p in enumerate(gt_files):
                        put(f'images/reference/gt/g{i}.png', p.read_bytes())
                    std = run / 'eval_samples_ctrl/eval200fix_input_g'
                    for i in range(187):
                        p = std / f'g{i}.png'
                        if not p.exists():
                            raise FileNotFoundError(p)
                        put(f'images/reference/std/g{i}.png', p.read_bytes())
                    common_written = True
            registry.append(item)
            print('COLLECTED', model_id, len(files), flush=True)

        if host == '48':
            base = 'assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow'
            for step in [35000, 40000, 45000, 50000, 55000]:
                add(f'base_{step//1000}k_ema', f'U100 Base {step//1000}k EMA', base, step, family='Base')
            add('base_55k_raw', 'U100 Base 55k RAW', 'assets/results/cfg_sweep/Baseline_55k_cfg1.0',
                55000, state='RAW', family='Base', training_run=base)
            deep = 'assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow'
            for step in [55000, 70000, 85000]:
                add(f'deep_{step//1000}k', f'Deep 114M {step//1000}k', deep, step, family='Deep')
            for slug, path in [('warm_gate', 'B_residual_gate'), ('warm_adapter', 'C_stage_cond_adapter'),
                               ('warm_independent', 'C_stage_cond_independent')]:
                candidates = sorted((root / 'assets/results/unet_dic' / path).glob('*-Unet-flow'))
                for step in [65000, 75000]:
                    for p in candidates:
                        add(f'{slug}_{step//1000}k', f'{slug} {step//1000}k', str(p), step, family='Warm ablation')
            for name in ['eval200_fixed.csv', 'train_top10_aug_sym.csv']:
                p = root / 'exp-std/csv' / name
                put(f'evidence/dataset/{name}', p.read_bytes())
            for name in ['model.py', 'blocks.py', 'io.py']:
                p = root / 'src/unet' / name
                put(f'evidence/current_source_48/{name}', p.read_bytes())
            p = root / 'tools/analysis/run_cfg_sweep_sampling.py'
            if p.exists():
                put('evidence/current_source_48/run_cfg_sweep_sampling.py', p.read_bytes())
        else:
            parent = 'assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow'
            for step in [20000, 50000, 60000, 70000, 75000, 80000]:
                for cfg in [1.0, 1.6]:
                    path = f'assets/results/eval_dic768_progression/step{step:07d}_cfg{cfg:.1f}'
                    add(f'dic_{step//1000}k_cfg{cfg:.1f}', f'DiC768 {step//1000}k CFG{cfg:.1f}',
                        path, step, cfg=cfg, family='DiC768', training_run=parent)
            for slug, path, run in [
                ('scratch_gate', 'arm1_resgate_b768', '20261019-023142-Unet-flow'),
                ('scratch_writer', 'arm2_writer_film_dec_b768', '20261019-105029-Unet-flow')]:
                add(f'{slug}_50k', f'{slug} 50k', f'assets/results/unet_ablation_50k/{path}/{run}',
                    50000, family='Scratch ablation')
            for slug, path, run, step in [
                ('decA_50k', 'armA_writer_decoupled_50k', '20261019-191851-Unet-flow', 50000),
                ('decA_70k', 'armA_writer_decoupled_70k', '20261020-065018-Unet-flow', 70000),
                ('decB_5k', 'armB_writer_decoupled_std_70k', '20261020-102201-Unet-flow', 5000),
                ('decB_10k', 'armB_writer_decoupled_std_70k', '20261020-102201-Unet-flow', 10000)]:
                add(slug, slug, f'assets/results/unet_decoupled_114m/{path}/{run}', step, family='Decoupled')
            parent = 'assets/results/unet_writer_distill_30k/20261018-202338-Unet-distill'
            for step in [5000, 10000, 15000]:
                for cfg in [1.0, 1.6]:
                    add(f'distill_{step//1000}k_cfg{cfg:.1f}', f'Distill {step//1000}k CFG{cfg:.1f}',
                        f'{parent}/eval_step{step:07d}_cfg{cfg:.1f}', step, cfg=cfg,
                        family='Distill', training_run=parent)
        put(f'evidence/registry_{host}.json', json.dumps(registry, ensure_ascii=False, indent=2).encode())
    print('ARCHIVE', output, Path(output).stat().st_size, 'bytes', len(registry), 'sets', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', choices=['48', '4090'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(args.root, args.host, args.output)
