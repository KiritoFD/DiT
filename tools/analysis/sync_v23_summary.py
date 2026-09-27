#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, pandas as pd

files = sorted(glob.glob('/root/Workspace/xy/DiT/assets/results/v23_splitnorm/eval_stdskel_summary.csv*'))
dfs = []
for f in files:
    try:
        d = pd.read_csv(f)
        if len(d) > 0 and 'step' in d.columns:
            dfs.append(d)
    except:
        pass
if dfs:
    full = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['step', 'set']).sort_values(['step', 'set'])
    target_p = '/root/Workspace/xy/DiT/assets/results/v23_splitnorm/eval_stdskel_summary.csv'
    full.to_csv(target_p, index=False)
    print(full[['step', 'set', 'ssim_mean', 'mse_mean', 'lpips_mean', 'ink_iou_mean', 'tgt_spec', 'cal_enrich']].to_string(index=False))
