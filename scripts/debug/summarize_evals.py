#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 5script/results 下所有实验的 eval 曲线到 CSV + 收集基模 grid 素材。
用法: python /tmp/summarize_evals.py
输出: /tmp/evals_summary.csv
      /tmp/grid_s30_130k/  /tmp/grid_v8a_95k/  (gt+sample 对)
"""
import os, json, glob, csv, sys, shutil

BASE = '/root/Workspace/xy/DiT'
os.chdir(BASE)
sys.stdout.reconfigure(encoding='utf-8')

OUT_CSV = '/tmp/evals_summary.csv'

def exp_name_from_dir(d):
    return os.path.basename(os.path.normpath(d))

rows = []
# 1) 基模实验: eval_auto_*.json (train.py daemon)
# 2) ctrl/repa 实验: eval_auto_ctrl_*.json (ctrl daemon)
for res_dir in ['s21_fame_flow_v2', 's26_ctrl_gt_skel', 's29_ctrl_gt_skel_1px',
                's30_dino_char_strong_pretrain', 's31_ctrl_gt_skel_1px',
                's32_repa_finetune', 's32b_repa_strong', 's32c_chain',
                'v8_3stage']:
    rd = os.path.join('5script/results', res_dir)
    if not os.path.isdir(rd):
        continue
    # 统一找所有含 checkpoints 的 run 目录 (支持 <res>/<run>/ 与 <res>/<stage>/<run>/ 两层嵌套)
    for ck in glob.glob(os.path.join(rd, '**', 'checkpoints'), recursive=True):
        if not os.path.isdir(ck):
            continue
        run = os.path.basename(os.path.dirname(ck))[:13]
        # 基模 (train.py)
        for f in sorted(glob.glob(os.path.join(ck, 'eval_auto_[0-9]*.json'))):
            try:
                j = json.load(open(f))
                step = int(os.path.basename(f).replace('eval_auto_','').replace('.json',''))
                rows.append({
                    'exp': res_dir, 'run': run[:13], 'kind': 'base',
                    'step': step,
                    'ssim': j.get('ssim',''), 'mse': j.get('mse',''),
                    'lpips': j.get('lpips',''), 'skel_iou': j.get('skel_iou','')})
            except Exception as e:
                print('skip', f, e)
        # ctrl/repa (ctrl daemon)
        for f in sorted(glob.glob(os.path.join(ck, 'eval_auto_ctrl_*.json'))):
            try:
                j = json.load(open(f))
                step = int(os.path.basename(f).replace('eval_auto_ctrl_','').replace('.json',''))
                c = j.get('ctrl', j); b = j.get('base', {})
                rows.append({
                    'exp': res_dir, 'run': run[:13], 'kind': 'ctrl',
                    'step': step,
                    'ssim': c.get('ssim',''), 'mse': c.get('mse',''),
                    'lpips': c.get('lpips',''), 'skel_iou': c.get('skel_iou',''),
                    'base_ssim': b.get('ssim','')})
            except Exception as e:
                print('skip', f, e)

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fo:
    w = csv.DictWriter(fo, fieldnames=['exp','run','kind','step','ssim','mse','lpips','skel_iou','base_ssim'])
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r['exp'], r['step'])):
        w.writerow(r)
print('wrote', OUT_CSV, 'rows:', len(rows))

# 每实验统计
from collections import defaultdict
per = defaultdict(list)
for r in rows:
    per[(r['exp'], r['kind'])].append(r)
for (e, k), rr in sorted(per.items()):
    best = max(rr, key=lambda x: float(x['ssim']) if x['ssim'] != '' else -1)
    print('%-28s %-5s n=%3d best_ssim=%s@%s' % (e, k, len(rr), best['ssim'], best['step']))

# ---- 基模 grid 素材: S30 old (130k) vs v8a (95k) ----
def pull_grid(src_dir, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    gts = sorted(glob.glob(os.path.join(src_dir, 'gt*.png')))[:n]
    got = 0
    for gt in gts:
        i = os.path.basename(gt)[2:-4]
        s = os.path.join(src_dir, 'sample%s.png' % i)
        if not os.path.exists(s):
            continue
        shutil.copy2(gt, os.path.join(out_dir, 'gt_%s.png' % i))
        shutil.copy2(s, os.path.join(out_dir, 'sample_%s.png' % i))
        got += 1
    print('grid', out_dir, got, 'pairs')

s30_src = '5script/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/eval_samples/step0130000'
v8_src = '5script/results/v8_3stage/v8a/20260902-125615-v8a-s30-base/checkpoints/eval_samples/step0095000'
pull_grid(s30_src, '/tmp/grid_s30_130k')
pull_grid(v8_src, '/tmp/grid_v8a_95k')
print('DONE')