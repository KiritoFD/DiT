#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成冒烟配置: v17_inj3_fixed_aug_100k @100k + w_style_rank=0.005, 续训 1k 步。

对照组 = 同 ckpt 同 1k 步、w_style_rank=0 (原地继续训)。
两份 config 唯一差别 = w_style_rank。判读三件套:
  Diff 稳定性 / strict(249) 不掉 / style-rank loss 走向。
"""
import json

BASE = 'src/train/configs/v17_inj3_fixed_aug_100k.json'
base = json.load(open(BASE, encoding='utf-8'))

for tag, w in (('styleA', 0.005), ('styleB', 0.0)):
    d = dict(base)
    d['experiment_name'] = 'smoke-%s' % tag
    d['results_dir'] = 'assets/results/smoke_%s' % tag
    d['max_steps'] = 101000          # 从 100k ckpt 续 1k 步
    d['ckpt_every'] = 500
    d['epoch_steps'] = 500           # 必须相等
    d['w_style_rank'] = w
    d['use_ema'] = False             # 短程冒烟要看裸权重
    d['in_mem_eval_sets'] = ('seen:assets/eval_v13_seen_fixed.csv:20,'
                             'strict:assets/eval_v13_strict_fixed.csv:249')
    d['_comment'] = ('冒烟 %s: v17_inj3_fixed_aug_100k@100k 续 1k 步, w_style_rank=%.4f '
                     '(t-gate [0.05,0.25], 冻结 2.9M 编码器 + 87 质心). 判据: Diff 稳定 / '
                     'strict 不掉 / rank loss 下降.' % (tag, w))
    out = 'src/train/configs/smoke_%s.json' % tag
    json.dump(d, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('wrote', out, 'w_style_rank =', w)