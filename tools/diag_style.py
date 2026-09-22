"""风格编码诊断：把「表塌缩是不是瓶颈」这件事用数据说清楚。

## 要回答的三个问题
1. 各实验的 style 表**余弦相似度**是多少？（表塌缩程度）
2. 各 ckpt 在**同一 step** 下的 strict 指标（新墨迹指标）谁好？
3. 书家级：strict 指标与"该书家样本数"的相关性 r 是多少？

## 输出
- assets/style_diag.json   全部诊断数据
- 终端打印摘要
"""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import numpy as np
import torch

os.chdir("/root/Workspace/xy/DiT")

OUT = {"tables": {}, "ckpts": [], "per_callig": {}}

# ── 1) 各实验的 style 表余弦 ────────────────────────────────────────
print("  === 1) style 表余弦（表塌缩程度）===")
for ck in sorted(glob.glob("assets/results/*/*/checkpoints/[0-9]*.pt"))[:400]:
    run = ck.split("/")[2]
    try:
        d = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception:
        continue
    sd = d.get("ema") or d.get("model") or d
    # 找 style/callig 相关的表
    for k, v in sd.items():
        kl = k.lower()
        if not any(s in kl for s in ("callig", "style", "basis")):
            continue
        if not hasattr(v, "shape") or v.ndim != 2:
            continue
        if v.shape[0] < 8:
            continue
        w = v.float()
        wn = w / (w.norm(dim=1, keepdim=True) + 1e-8)
        cos = (wn @ wn.T)
        n = cos.shape[0]
        off = (cos.sum() - cos.diag().sum()) / (n * (n - 1))
        OUT["tables"].setdefault(run, []).append({
            "key": k, "shape": list(w.shape),
            "mean_cos": round(float(off), 4),
            "std_cos": round(float(cos[~torch.eye(n, dtype=bool)].std()), 4),
            "eff_rank": round(float(torch.linalg.matrix_rank(w).item()), 1),
        })

for run, lst in sorted(OUT["tables"].items()):
    for t in lst[:2]:
        print(f"    {run:<28} {t['key'][:34]:<36} "
              f"shape={t['shape']}  cos={t['mean_cos']}  rank={t['eff_rank']}")

# ── 2) 各 ckpt 的 strict 指标（同 step 对比）────────────────────────
print("\n  === 2) 各 ckpt 的 strict（新墨迹指标）===")
for f in sorted(glob.glob("assets/ink_eval_raw/*__strict.csv")):
    run = os.path.basename(f).replace("__strict.csv", "")
    rs = list(csv.DictReader(open(f, encoding="utf-8")))
    if not rs:
        continue
    step = rs[0].get("step", "")

    def m(k):
        vs = [float(r[k]) for r in rs if r.get(k) not in ("", None)]
        return statistics.mean(vs) if vs else 0.0
    OUT["ckpts"].append({
        "run": run, "step": step, "n": len(rs),
        "ssim": round(m("ssim"), 4), "ink_ssim": round(m("ink_ssim"), 4),
        "ink_iou": round(m("ink_iou"), 4), "skel_iou": round(m("skel_iou"), 4),
        "lpips": round(m("lpips"), 4)})

OUT["ckpts"].sort(key=lambda x: -x["ink_iou"])
print(f"    {'run':<26}{'step':<9}{'ssim':<9}{'ink':<9}{'iou':<9}lpips")
for c in OUT["ckpts"]:
    print(f"    {c['run']:<26}{str(c['step']):<9}{c['ssim']:<9}"
          f"{c['ink_ssim']:<9}{c['ink_iou']:<9}{c['lpips']}")

# ── 3) 书家级：指标 vs 样本数 ──────────────────────────────────────
print("\n  === 3) 书家级：ink_iou vs 样本数 ===")
# 用训练集统计每个书家的样本数
cnt = defaultdict(int)
for r in csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                             encoding="utf-8")):
    cnt[r.get("calligrapher", "")] += 1
# 用某个 ckpt 的逐样本指标按书家聚合（CSV 里没书家列，用 idx 反查）
rows = list(csv.DictReader(open("assets/train_50k_v2_fixed.csv",
                                encoding="utf-8")))
per = defaultdict(list)
f = "assets/ink_eval_raw/v13_base_50k__strict.csv"
for r in csv.DictReader(open(f, encoding="utf-8")):
    i = int(r["idx"])
    if i >= len(rows):
        continue
    cg = rows[i].get("calligrapher", "")
    per[cg].append(float(r["ink_iou"]))
OUT["per_callig"] = {k: {"n_train": cnt.get(k, 0),
                         "n_eval": len(v),
                         "mean_iou": round(statistics.mean(v), 4)}
                     for k, v in per.items()}
xs = [v["n_train"] for v in OUT["per_callig"].values() if v["n_eval"] >= 2]
ys = [v["mean_iou"] for v in OUT["per_callig"].values() if v["n_eval"] >= 2]
if len(xs) >= 3:
    r_ = float(np.corrcoef(xs, ys)[0, 1])
    OUT["corr_n_vs_iou"] = round(r_, 3)
    print(f"    书家样本数 vs strict ink_iou 的相关性 r = {r_:.3f}")
    print(f"    （历史记录: 全图 SSIM 的 r = -0.62）")
    top = sorted(OUT["per_callig"].items(), key=lambda x: x[1]["mean_iou"])
    print(f"\n    {'书家':<12}{'训练数':<8}{'eval数':<8}ink_iou")
    for k, v in top[:8]:
        print(f"    {k:<12}{v['n_train']:<8}{v['n_eval']:<8}{v['mean_iou']}")
    print("    ...")
    for k, v in top[-4:]:
        print(f"    {k:<12}{v['n_train']:<8}{v['n_eval']:<8}{v['mean_iou']}")

with open("assets/style_diag.json", "w", encoding="utf-8") as fp:
    json.dump(OUT, fp, ensure_ascii=False, indent=1)
print("\n  -> assets/style_diag.json")
