# -*- coding: utf-8 -*-
import csv

rows = [r for r in csv.DictReader(open("assets/results/v11_pretrain_M432_adaln4_sym_noise400k/eval_stdskel_summary.csv", encoding="utf-8")) if r["set"] == "strict"]
best = max(rows, key=lambda r: float(r["ssim_mean"]))
print("best strict:", best["step"], best["ssim_mean"])
print("recent:", [(r["step"], r["ssim_mean"]) for r in rows[-8:]])
s = [r for r in csv.DictReader(open("assets/results/v11_pretrain_M432_adaln4_sym_noise400k/eval_stdskel_summary.csv", encoding="utf-8")) if r["set"] == "seen"]
bs = max(s, key=lambda r: float(r["ssim_mean"]))
print("best seen:", bs["step"], bs["ssim_mean"])
