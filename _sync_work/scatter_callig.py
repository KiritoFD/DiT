"""画「书家风格作用强度」 vs 「该书家 strict ssim」散点。

用途：区分 strict 停滞的瓶颈在【数据】还是【条件机制】
  沿对角线（斜率大）-> 机制没问题，是数据
  水平线（斜率≈0） -> 机制无区分度 -> 改注入
  竖线             -> 风格有效但对泛化没帮助 -> 瓶颈在别处

X 来源: assets/diversity_inter_pairs_v13_155k_percalig.csv
  ⚠ 该文件有**两种列名格式**（第一版把两列写反了），自动识别：
      新版: char_a, callig_a, char_b, callig_b
      旧版: callig_a(=字), glyph_a(=书家), ...
Y 来源: <results>/eval_stdskel_batch.csv (set=strict, idx, ssim, calligrapher)
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

os.chdir("/root/Workspace/xy/DiT")

PAIRS = "assets/diversity_inter_pairs_v13_155k_percalig.csv"
pairs = list(csv.DictReader(open(PAIRS, encoding="utf-8")))
cols = set(pairs[0].keys())
new_fmt = "char_a" in cols
print(f"  pairs 格式: {'新版 (char_a/callig_a)' if new_fmt else '旧版 (列名反了)'}")

ka = "callig_a" if new_fmt else "glyph_a"
kb = "callig_b" if new_fmt else "glyph_b"
by_cal = defaultdict(list)
for r in pairs:
    if r["kind"] != "callig":
        continue
    d = 1.0 - float(r["ssim"])
    by_cal[r[ka]].append(d)
    by_cal[r[kb]].append(d)
X = {c: float(np.mean(v)) for c, v in by_cal.items()}
print(f"\n  === X: 书家风格作用强度（{len(X)} 个书家，{len(pairs)} 对）===")
for c, v in sorted(X.items(), key=lambda kv: -kv[1]):
    print(f"    {c:<8} {v:.4f}  (n={len(by_cal[c])})")

# ⚠ 必须**只认 v13 base** —— 用 assets/results/*/ 的 glob 会挑到 v10b 等别的实验，
#   那样 X 和 Y 来自不同模型，散点毫无意义（实测踩过：挑中了 v10b step390000）。
cands = sorted(glob.glob("assets/results/v13_base_50k/eval_stdskel_batch.csv"))
Y, step_used, src = {}, None, None
for p in cands:
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["set"] == "strict"]
    if not rows:
        continue
    st = max(int(r["step"]) for r in rows)
    rows = [r for r in rows if int(r["step"]) == st]
    if not any(r.get("calligrapher") for r in rows):
        continue
    by = defaultdict(list)
    for r in rows:
        by[r["calligrapher"]].append(float(r["ssim"]))
    if len(rows) > sum(len(v) for v in Y.values()):
        Y = {c: float(np.mean(v)) for c, v in by.items()}
        step_used, src = st, p

if not Y:
    sr = list(csv.DictReader(open("assets/eval_v13_strict.csv", encoding="utf-8")))
    for p in cands:
        rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
                if r["set"] == "strict"]
        if not rows:
            continue
        st = max(int(r["step"]) for r in rows)
        rows = sorted([r for r in rows if int(r["step"]) == st],
                      key=lambda r: int(r["idx"]))
        by = defaultdict(list)
        for r in rows:
            i = int(r["idx"])
            if i < len(sr):
                by[sr[i]["calligrapher"]].append(float(r["ssim"]))
        if len(by) > len(Y):
            Y = {c: float(np.mean(v)) for c, v in by.items()}
            step_used, src = st, p + " (idx 对齐)"
print(f"\n  === Y: 该书家 strict ssim（step={step_used}，{len(Y)} 个书家）===")
print(f"     来源: {src}")

common = sorted(set(X) & set(Y))
print(f"\n  === 两边都有: {len(common)} 个书家 ===")
if common:
    print(f"  {'书家':<10}{'风格强度':>10}{'strict':>10}{'n':>5}")
    xs, ys, ns = [], [], []
    for c in common:
        n = len(by_cal.get(c, []))
        print(f"  {c:<10}{X[c]:>10.4f}{Y[c]:>10.4f}{n:>5}")
        xs.append(X[c]); ys.append(Y[c]); ns.append(n)
    xs_, ys_ = np.array(xs), np.array(ys)
    if len(xs_) >= 3:
        r = float(np.corrcoef(xs_, ys_)[0, 1])
        k, b = np.polyfit(xs_, ys_, 1)
        print(f"\n  Pearson r = {r:+.4f}")
        print(f"  线性拟合: strict = {k:+.3f} * style {b:+.3f}")
        if k > 0.3:
            v = "沿对角线 -> 机制没问题，瓶颈在【数据】"
        elif abs(k) < 0.15:
            v = "接近水平 -> 机制无区分度，该改【注入】"
        else:
            v = "弱正相关 -> 两者都有影响"
        print(f"  判读: {v}")
    with open("assets/scatter_callig_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["callig", "style_strength", "strict_ssim", "n_pairs"])
        for c, x, y, n in zip(common, xs, ys, ns):
            w.writerow([c, round(x, 6), round(y, 6), n])
    print("  written assets/scatter_callig_data.csv")
else:
    print("  ⚠ 没有交集")
    print(f"    X 样例: {sorted(X)[:6]}")
    print(f"    Y 样例: {sorted(Y)[:6]}")
