#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_ablation_matrix.py —— 顺序执行 M0~M4 五组消融实验（每组 5,000 steps）并统一产出对比报告。

消融矩阵定义：
- M0: 现有 D1 (haar='none', metric='ranking')
- M1: D1 + 论文绝对距离 Contrastive (haar='none', metric='contrastive')
- M2: D1 + 1D Haar (haar='1d', metric='ranking')
- M3: D1 + 2D Haar (haar='2d', metric='ranking')
- M4: D1 + 2D Haar + 论文绝对距离 Contrastive (haar='2d', metric='contrastive')
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ARMS = ['M0', 'M1', 'M2', 'M3', 'M4']
DESCRIPTIONS = {
    'M0': 'D1 Baseline (No Haar, Cosine Ranking Softplus)',
    'M1': 'D1 + Paper Euclidean Contrastive (m=1.0)',
    'M2': 'D1 + 1D Global Vector Haar + Ranking',
    'M3': 'D1 + 2D Spatial Haar DWT + Ranking',
    'M4': 'D1 + 2D Spatial Haar DWT + Paper Contrastive'
}


def run_command(cmd, log_file=None):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUNNING: {' '.join(cmd)}", flush=True)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                f.write(line)
                f.flush()
                # Print periodic progress to console
                if 'TRAIN' in line or 'VALIDATION' in line or 'PIXEL_TRAIN' in line or 'PIXEL_ACCEPTANCE' in line:
                    print(f"  {line.strip()}", flush=True)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='scratch/writer_pixel_d0d1_20261016_01/assets/data')
    p.add_argument('--out-dir', default='assets/runs/ablation_5k')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--quartet-samples', type=int, default=384)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--python', default='/opt/conda/envs/cu121/bin/python')
    args = p.parse_args()

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    master_log = base_out / 'ablation_runner.log'

    print("=" * 80)
    print(f"Starting Multi-loss Siamese CNN Ablation Matrix (M0 ~ M4)")
    print(f"Steps per arm: {args.steps} | Effective Batch: {args.batch * args.grad_accum}")
    print(f"Output directory: {base_out}")
    print("=" * 80, flush=True)

    summary = {}

    for arm in ARMS:
        run_dir = base_out / arm
        run_log = run_dir / 'train.log'
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> [Stage {arm}] {DESCRIPTIONS[arm]} Starting...", flush=True)
        t0 = time.time()

        # 1. Train 5000 steps
        train_cmd = [
            args.python, 'tools/train_writer_style_pixel.py',
            '--data-root', args.data_root,
            '--out', str(run_dir),
            '--arm', arm,
            '--batch', str(args.batch),
            '--quartet-samples', str(args.quartet_samples),
            '--grad-accum', str(args.grad_accum),
            '--steps', str(args.steps),
            '--eval-every', str(args.eval_every)
        ]
        run_command(train_cmd, log_file=run_log)

        # 2. Run Acceptance Test
        print(f">>> [Stage {arm}] Running Acceptance Evaluation...", flush=True)
        accept_cmd = [
            args.python, 'tools/analysis/accept_writer_style_pixel.py',
            '--run', str(run_dir),
            '--data-root', args.data_root
        ]
        run_command(accept_cmd, log_file=run_log)

        # 3. Read acceptance.json
        accept_json = run_dir / 'acceptance.json'
        if accept_json.exists():
            with open(accept_json, encoding='utf-8') as f:
                res = json.load(f)
            clean = res['conditions_evaluated']['clean']
            bg = res['conditions_evaluated']['background_only']
            summary[arm] = {
                'description': DESCRIPTIONS[arm],
                'elapsed_sec': round(time.time() - t0, 1),
                'best_step': res.get('step', args.steps),
                'test_char_matched_acc': clean['test_char']['matched']['by_char']['accuracy'],
                'test_char_matched_ci95': clean['test_char']['matched']['by_char']['ci95'],
                'test_char_map': clean['test_char']['crosschar_map'],
                'test_source_matched_acc': clean['test_source']['matched']['by_char']['accuracy'],
                'test_source_matched_ci95': clean['test_source']['matched']['by_char']['ci95'],
                'test_source_map': clean['test_source']['crosschar_map'],
                'image_disjoint_source_acc': clean['test_source']['matched']['image_disjoint']['accuracy'],
                'bg_probe_train_ce': bg['train_slot_ce_top1'],
                'bg_probe_matched_acc': bg['test_char']['matched']['by_char']['accuracy']
            }
            # Save rolling summary
            with open(base_out / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f">>> [Stage {arm}] Completed in {(time.time() - t0) / 60:.1f} mins.", flush=True)

    print("\n" + "=" * 80)
    print("ALL ABLATIONS COMPLETED! FINAL SUMMARY:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
