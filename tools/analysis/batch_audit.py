# -*- coding: utf-8 -*-
"""batch_audit.py — 口径审计：把 batch × steps = 总样本数（与 epoch）一次列清

⚠ 教训: 只比 steps 而不比 **batch** → 在不同 batch 的臂之间会得出错误结论
   （本仓 48 的消融/增广臂 batch=512, 而主线 v68/v81 batch=288 → 匹配步数下样本数差 1.78×）。
   以后所有"匹配步数"的说法必须同时给出 **总样本数**。
"""
import glob
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PATTERNS = [
    ("主线/本机", ["src/train/configs/v6*.json", "src/train/configs/v7*.json", "src/train/configs/v8*.json"]),
    ("消融(48)", ["src/train/configs/ablation/*.json"]),
    ("增广(48)", ["src/train/configs/*aug*noOT*.json", "src/train/configs/*morphE*.json",
                  "src/train/configs/*elastic*.json", "src/train/configs/*grid*.json",
                  "src/train/configs/*affine*.json"]),
]
rows, seen = [], set()
for group, pats in PATTERNS:
    files = []
    for p in pats:
        files += glob.glob(p)
    for f in sorted(set(files)):
        if f in seen:
            continue
        seen.add(f)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        b, s = d.get("global_batch_size") or d.get("batch_size"), d.get("max_steps")
        if not b or not s:
            continue
        csvp = str(d.get("data_csv", "") or "")
        nrows = None
        if csvp and os.path.exists(csvp):
            try:
                with open(csvp, encoding="utf-8") as fh:
                    nrows = sum(1 for _ in fh) - 1
            except Exception:
                pass
        tot = b * s
        rows.append((group, os.path.basename(f), b, s, tot, nrows,
                     (tot / nrows if nrows else None), d.get("use_c2ot"), d.get("std_soft_cond")))

print(f"  {'组':<9}{'配置':<44}{'batch':>6}{'steps':>8}{'总样本':>9}{'csv行':>8}{'epoch':>7}  c2ot stdsoft")
for g, name, b, s, tot, nrows, ep, c2, ss in rows:
    print(f"  {g:<9}{name[:43]:<44}{b:>6}{s:>8}{tot/1e6:>8.1f}M{(nrows or 0):>8}{(ep or 0):>7.0f}  "
          f"{str(c2):<5} {str(ss)}")

print("""
  读法（必须遵守）:
   * **匹配步数 ≠ 匹配样本**。batch 不同时, 同 step 的样本数差 batch 之比。
   * 主线 v68/v81: batch 288 → 匹配步数即匹配样本 ✓
   * 48 消融/增广: batch 512 → 要 288 的臂匹配它们的 51.2M 样本, 需 512*100000/288 = 177,778 步
""")
