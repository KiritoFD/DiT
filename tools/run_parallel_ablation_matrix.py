#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_parallel_ablation_matrix.py —— 4路并发执行 M0~M4 消融实验（把显存吃到 20G，全速并行）。

消融矩阵定义：
- M0: 现有 D1 (none, ranking)
- M1: D1 + 论文 Contrastive (none, contrastive)
- M2: D1 + 1D Haar + Ranking (1d, ranking)
- M3: D1 + 2D Haar + Ranking (2d, ranking)
- M4: D1 + 2D Haar + Contrastive (2d, contrastive)
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
    'M0': 'D1 Baseline (No Haar, Ranking Softplus)',
    'M1': 'D1 + Paper Euclidean Contrastive (m=1.0)',
    'M2': 'D1 + 1D Global Vector Haar + Ranking',
    'M3': 'D1 + 2D Spatial Haar DWT + Ranking',
    'M4': 'D1 + 2D Spatial Haar DWT + Paper Contrastive'
}


def run_acceptance(arm, run_dir, data_root, python_bin):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running acceptance for {arm}...", flush=True)
    cmd = [python_bin, 'tools/analysis/accept_writer_style_pixel.py', '--run', str(run_dir), '--data-root', str(data_root)]
    subprocess.run(cmd, check=True)
    
    accept_json = run_dir / 'acceptance.json'
    if accept_json.exists():
        with open(accept_json, encoding='utf-8') as f:
            res = json.load(f)
        clean = res['conditions_evaluated']['clean']
        bg = res['conditions_evaluated']['background_only']
        return {
            'description': DESCRIPTIONS[arm],
            'best_step': res.get('step', 5000),
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
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='scratch/writer_pixel_d0d1_20261016_01/assets/data')
    p.add_argument('--out-dir', default='assets/runs/ablation_5k')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--quartet-samples', type=int, default=384)
    p.add_argument('--grad-accum', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--python', default='/opt/conda/envs/cu121/bin/python')
    args = p.parse_args()

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    summary_file = base_out / 'summary.json'

    # Load existing summary if available
    summary = {}
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text(encoding='utf-8'))
        except Exception:
            pass

    print("=" * 80)
    print(f"Parallel Multi-loss Siamese CNN Ablation Runner (Max Concurrent: {args.max_workers})")
    print(f"Target VRAM usage: ~20GB | Steps per arm: {args.steps} | Effective Batch: {args.batch * args.grad_accum}")
    print(f"Queue: {ARMS}")
    print("=" * 80, flush=True)

    pending = list(ARMS)
    active = {}  # arm -> (proc, log_f, start_time)

    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    while pending or active:
        # Launch new jobs if worker slots available
        while pending and len(active) < args.max_workers:
            arm = pending.pop(0)
            run_dir = base_out / arm
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / 'train.log'
            log_f = open(log_path, 'a', encoding='utf-8')

            cmd = [
                args.python, 'tools/train_writer_style_pixel.py',
                '--data-root', str(args.data_root),
                '--out', str(run_dir),
                '--arm', arm,
                '--batch', str(args.batch),
                '--quartet-samples', str(args.quartet_samples),
                '--grad-accum', str(args.grad_accum),
                '--steps', str(args.steps),
                '--eval-every', str(args.eval_every)
            ]

            # Resume if last.pt exists
            last_pt = run_dir / 'last.pt'
            if last_pt.exists():
                cmd.extend(['--resume', str(last_pt)])
                print(f"[{time.strftime('%H:%M:%S')}] Resuming [{arm}] from {last_pt}...", flush=True)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Starting [{arm}] {DESCRIPTIONS[arm]}...", flush=True)

            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            active[arm] = (proc, log_f, time.time())
            time.sleep(2)  # Stagger process startup slightly

        # Check running processes
        time.sleep(5)
        done_arms = []
        for arm, (proc, log_f, start_t) in active.items():
            ret = proc.poll()
            if ret is not None:
                log_f.close()
                done_arms.append(arm)
                elapsed = time.time() - start_t
                if ret != 0:
                    print(f"[{time.strftime('%H:%M:%S')}] ERROR: [{arm}] failed with exit code {ret}!", flush=True)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] SUCCESS: [{arm}] completed in {elapsed / 60:.1f} mins.", flush=True)
                    res = run_acceptance(arm, base_out / arm, args.data_root, args.python)
                    if res:
                        summary[arm] = res
                        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(f"[{time.strftime('%H:%M:%S')}] Updated summary saved to {summary_file}", flush=True)

        for arm in done_arms:
            del active[arm]

    print("\n" + "=" * 80)
    print("ALL ABLATIONS FINISHED! Final summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
