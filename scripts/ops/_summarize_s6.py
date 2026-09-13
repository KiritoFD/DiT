#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, json, os

BASE = "/root/Workspace/xy/DiT/5script/results/"
EXPS = ["s6_top6_diffonly", "s6_top6_struct_fp32", "s6_top6_struct_fp32_full", "s6_top6_diff_then_struct"]
for e in EXPS:
    jsons = sorted(glob.glob(os.path.join(BASE, e, "*/checkpoints/eval_auto_*.json")))
    print(f"=== {e}: {len(jsons)} eval_auto ===", flush=True)
    if not jsons:
        continue
    for jp in jsons[-5:]:
        try:
            d = json.load(open(jp))
            print(f"  step {d.get('step')}: mse={d.get('mse'):.4f} ssim={d.get('ssim'):.4f}", flush=True)
        except Exception as ex:
            print(f"  {os.path.basename(jp)} ERR {ex}", flush=True)
    # 训练日志最后一行(看结束方式: epoch 耗尽 / 早停 / 仍在跑)
    logs = glob.glob(os.path.join(BASE, e, "*/log.txt"))
    if logs:
        lines = open(logs[-1], encoding="utf-8", errors="replace").read().splitlines()
        for ln in lines[-2:]:
            if "step=" in ln or "Reached" in ln or "early" in ln.lower() or "Done" in ln or "epoch" in ln:
                print(f"  LOG: {ln[-110:]}", flush=True)