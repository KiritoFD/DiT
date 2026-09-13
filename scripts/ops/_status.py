#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, json, os

# exp1 resume 最新 eval 曲线
pat = "/root/Workspace/xy/DiT/5script/results/s6_top6_diffonly/*resume*/checkpoints/eval_auto_*.json"
files = sorted(glob.glob(pat))
print("=== exp1 diffonly resume eval ===", flush=True)
for jp in files:
    try:
        d = json.load(open(jp))
        print(f"  step {d.get('step')}: mse={d.get('mse'):.4f} ssim={d.get('ssim'):.4f}", flush=True)
    except Exception as ex:
        print(jp, "ERR", ex, flush=True)

# 当前训练 step (最新 log 行)
logp = max(glob.glob("/root/Workspace/xy/DiT/5script/results/s6_top6_diffonly/*resume*/log.txt"), default=None, key=os.path.getmtime)
if logp:
    lines = open(logp, encoding="utf-8", errors="replace").read().splitlines()
    for ln in lines[-3:]:
        print(f"  LOG: {ln[-120:]}", flush=True)

# run_s6_resume.log 看序列进展
rp = "/root/Workspace/xy/DiT/run_s6_resume.log"
if os.path.exists(rp):
    print("=== run_s6_resume.log tail ===", flush=True)
    for ln in open(rp, encoding="utf-8", errors="replace").read().splitlines()[-5:]:
        print(f"  {ln}", flush=True)