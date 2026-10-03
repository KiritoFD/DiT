#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v18_style_rank_200k 预训练配置（从头，200k cosine，挂 style-rank loss）。

基模 = v17_inj3_fixed_aug_100k 全配方（inj3 三件套 + 条件增强 + fixed 数据），
唯一新增 = w_style_rank=0.005（冻结 2.9M 编码器 + 87 质心, t∈[0.05,0.25]）。
判据: strict 轨迹 vs v17_inj3_fixed_aug_100k@100k (0.5454) 同口径 + 探针富集
      + ratio_style（跑完后用 tools/eval_diversity.py）。
"""
import json

BASE = 'src/train/configs/v17_inj3_fixed_aug_100k.json'
b = json.load(open(BASE, encoding='utf-8'))

d = dict(b)
d['experiment_name'] = 'v18-style-rank-200k'
d['results_dir'] = 'assets/results/v18_style_rank_200k'
d['max_steps'] = 200000
d['lr'] = 0.0005
d['lr_schedule'] = 'cosine'
d['warmup_steps'] = 3000
d['min_lr_ratio'] = 0.1
d['w_style_rank'] = 0.005
d['style_rank_t_min'] = 0.05
d['style_rank_t_max'] = 0.25
d['style_rank_ckpt'] = 'assets/style_enc_latent.pt'
d['style_rank_cent'] = 'assets/rank_cent87.npy'
d['use_ema'] = True
d['ckpt_every'] = 5000
d['epoch_steps'] = 5000
d['_comment'] = (
    'v18 预训练（2026-09-25）: v17_inj3_fixed_aug 全配方（骨架 0.15 丢弃 / 8ch 拼接 / '
    '第 6 层 LocalCA / 离线几何 4 档 + 在线数值增强 / fixed 数据）+ '
    '**style-rank loss w=0.005**（冻结 2.9M latent 编码器 + 87 pair 质心, t∈[0.05,0.25] 门控, '
    'loss=1-cos(f(pred_x0), centroid[y_pair]), added=0）。'
    '冒烟(2026-09-25 05:2x)已过: 1k 步 Diff 0.275->0.268 稳定下降、strict@100.5k 0.5481 '
    '(vs 基模 100k 0.5454)、seen 0.5801、无发散。'
    '判据: strict 轨迹 vs v17 基模同口径 + cal_enrich 探针 + 完赛后 ratio_style '
    '(baseline: v13_base 1.48 / v15a 1.21, 同协议)。')


def main():
    out = 'src/train/configs/v18_style_rank_200k.json'
    json.dump(d, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('wrote', out)
    for k in ('max_steps', 'lr', 'w_style_rank', 'style_rank_t_min', 'style_rank_t_max',
              'use_ema', 'global_batch_size'):
        print(' ', k, '=', d[k])


if __name__ == '__main__':
    main()