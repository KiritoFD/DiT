#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线补算历史 run 的断笔比（frag_ratio）—— 不占 GPU。

动机: `frag_ratio` 只在新版 `in_mem_eval` 里算, E0/E1/gate/midskel20 的逐样本 CSV 里
全是 nan -> 用户的主判据「断笔比」**没有 baseline**。
但 in_mem_eval 把每个 step 的生成图/真迹都存成了 `eval_samples_ctrl/step*/{set}/g{i}.png|gt{i}.png`
-> 可以事后在 CPU 上把指标全部补算出来。

用法: python _sync_work/backfill_frag.py <results_dir> [--out <csv>]
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.metrics import frag_ratio, hole_ratio, ssim  # noqa: E402
from src.eval.metrics_ink import ink_iou, ink_ssim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('results_dir')
ap.add_argument('--out', default='')
a = ap.parse_args()

out = a.out or os.path.join(a.results_dir, 'eval_backfill_frag.csv')
raw_out = out.replace('.csv', '_per_sample.csv')
samples_root = os.path.join(a.results_dir, 'eval_samples_ctrl')
if not os.path.isdir(samples_root):
    raise SystemExit(f'✗ 没有 {samples_root} —— 该 run 没存逐样本图')

from PIL import Image  # noqa: E402

rows_out = []
raw_rows = []
step_dirs = sorted(glob.glob(os.path.join(samples_root, 'step*')))
print(f'找到 {len(step_dirs)} 个 step 目录')
for sd in step_dirs:
    m = re.search(r'step(\d+)', os.path.basename(sd))
    if not m:
        continue
    step = int(m.group(1))
    for setname in sorted(os.listdir(sd)):
        d = os.path.join(sd, setname)
        if not os.path.isdir(d):
            continue
        gs = sorted(glob.glob(os.path.join(d, 'g*.png')))
        gs = [p for p in gs if not os.path.basename(p).startswith('gt')]
        if not gs:
            continue
        frags, holes_p, holes_g, inks, inkious, ssims = [], [], [], [], [], []
        for gp in gs:
            idx = os.path.basename(gp)[1:-4]
            gtp = os.path.join(d, f'gt{idx}.png')
            if not os.path.exists(gtp):
                continue
            g = np.asarray(Image.open(gp).convert('RGB'), dtype=np.float32) / 255.0
            t = np.asarray(Image.open(gtp).convert('RGB'), dtype=np.float32) / 255.0
            fr = frag_ratio(g, t, thresh=0.5)
            hp = hole_ratio(g, thresh=0.5)
            hg = hole_ratio(t, thresh=0.5)
            ik = ink_ssim(g, t)
            ii = ink_iou(g, t)
            ss = ssim(g, t)
            frags.append(fr)
            holes_p.append(hp)
            holes_g.append(hg)
            inks.append(ik)
            inkious.append(ii)
            ssims.append(ss)
            # ★ 逐样本行: 口径断裂时(旧集 vs 修复集)必须靠 subset 才可比
            raw_rows.append({
                'step': step, 'set': setname, 'idx': int(idx),
                'frag_ratio': fr, 'hole_pred': hp, 'hole_gt': hg,
                'ink_ssim': ik, 'ink_iou': ii, 'ssim': ss,
            })
        if not frags:
            continue
        rows_out.append({
            'step': step, 'set': setname, 'n': len(frags),
            'frag_mean': float(np.mean(frags)),
            'hole_pred_mean': float(np.mean(holes_p)),
            'hole_gt_mean': float(np.mean(holes_g)),
            'ink_ssim_mean': float(np.mean(inks)),
            'ink_iou_mean': float(np.mean(inkious)),
            'ssim_mean': float(np.mean(ssims)),
        })
        print(f'  step{step} {setname}: n={len(frags)} frag={np.mean(frags):.3f} '
              f'ink_ssim={np.mean(inks):.4f} ssim={np.mean(ssims):.4f}', flush=True)

with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
print(f'-> {out}  ({len(rows_out)} 行)')
if raw_rows:
    with open(raw_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)
    print(f'-> {raw_out}  ({len(raw_rows)} 行)')
