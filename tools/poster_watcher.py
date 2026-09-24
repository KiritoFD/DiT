#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster watcher：独立进程，轮询新 step 就重画 poster。

为什么需要: `render_poster` 是在**训练进程内**被调用的，改它必须重启训练才生效
（已经为这个重启过两次）。watcher 把"重画 poster"从训练进程里解耦出来，
以后改 poster 只重启 watcher 即可。

它做的是**同样的事**、用**同一份配置**，但用的是磁盘上最新的代码。
训练自己那次渲染会被 watcher 在下一次轮询时覆盖掉（内容相同或更新）。

用法:
  nohup python tools/poster_watcher.py <results_dir> > /tmp/poster_watch.log 2>&1 &
"""
import argparse
import glob
import json
import os
import re
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

ap = argparse.ArgumentParser()
ap.add_argument('results_dir')
ap.add_argument('--interval', type=int, default=30)
ap.add_argument('--settle', type=int, default=8, help='发现新 step 后等几秒让图写完')
ap.add_argument('--once', action='store_true', help='只跑一轮（调试用）')
a = ap.parse_args()

rd = a.results_dir.rstrip('/')
cf = sorted(glob.glob(os.path.join(rd, '*', 'resolved_config.json')))
if not cf:
    raise SystemExit(f'✗ 找不到 {rd}/*/resolved_config.json')
cfg = json.load(open(cf[-1], encoding='utf-8'))
SETS = []
for spec in str(cfg.get('in_mem_eval_sets', '')).split(','):
    bits = spec.split(':')
    if len(bits) == 3:
        SETS.append((bits[0], bits[1], int(bits[2])))
TRAIN_CSV = cfg.get('data_csv')
print(f'[watch] {rd}')
print(f'[watch] sets={[(s[0], s[2]) for s in SETS]}  train_csv={TRAIN_CSV}', flush=True)


def signature():
    """当前所有 step 目录 + 各 set 的列数。变了就说明有新 eval。"""
    out = []
    for d in sorted(glob.glob(os.path.join(rd, 'eval_samples_ctrl', 'step*'))):
        m = re.search(r'step(\d+)', os.path.basename(d))
        if not m:
            continue
        ns = []
        for nm, _c, _n in SETS:
            sub = 'g' if nm in ('seen', 'g') else nm
            n = 0
            while os.path.exists(os.path.join(d, sub, f'g{n}.png')):
                n += 1
            ns.append(n)
        out.append((int(m.group(1)), tuple(ns)))
    return tuple(out)


last = None
while True:
    try:
        sig = signature()
        if sig != last:
            if last is not None:
                time.sleep(a.settle)          # 让训练那边把图写完
                sig = signature()
            t0 = time.time()
            from src.eval.in_mem_eval import render_poster
            for nm, csvp, n in SETS:
                try:
                    p = render_poster(rd, nm, train_csv=TRAIN_CSV,
                                      eval_csv=csvp, n_eval=n)
                    print(f'[watch] {time.strftime("%H:%M:%S")} {nm} -> {p} '
                          f'({time.time() - t0:.1f}s)', flush=True)
                except Exception as e:
                    print(f'[watch] ✗ {nm} render failed: {e!r}', flush=True)
            last = sig
    except Exception as e:
        print(f'[watch] ✗ loop error: {e!r}', flush=True)
    if a.once:
        break
    time.sleep(a.interval)
