"""找出「在所有 ckpt 上 SSIM 都低」的样本 —— 这类样本本身很可能有问题。

## 原理
如果某条样本在**所有 ckpt** 上 SSIM 都很低，那不是模型的问题，
而是**这条样本本身有问题**（GT 与 std g 不匹配、图坏了、字太罕见等）。

## 输出
  assets/eval_low_ssim.csv  逐样本的跨 ckpt 统计（按 mean_ssim 升序）
"""
import csv
import glob
import os
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")

files = sorted(glob.glob("assets/full_eval_raw/*__strict.csv"))
print(f"  {len(files)} 个 strict raw 文件")

# 聚合：img_id -> [ssim, ...]
acc = defaultdict(list)
meta = {}
for f in files:
    exp = os.path.basename(f).replace("__strict.csv", "")
    for r in csv.DictReader(open(f, encoding="utf-8")):
        k = r["img_id"]
        try:
            acc[k].append(float(r["ssim"]))
        except ValueError:
            continue
        if k not in meta:
            meta[k] = {"char": r.get("char", ""), "script": r.get("script", ""),
                       "idx": r.get("idx", "")}

print(f"  聚合到 {len(acc)} 个唯一样本")

rows = []
for k, vs in acc.items():
    if len(vs) < 3:                 # 至少 3 个 ckpt 评过
        continue
    m = sum(vs) / len(vs)
    rows.append({"img_id": k, "n_ckpt": len(vs),
                 "mean_ssim": round(m, 4),
                 "min_ssim": round(min(vs), 4),
                 "max_ssim": round(max(vs), 4),
                 "char": meta[k]["char"], "script": meta[k]["script"],
                 "idx": meta[k]["idx"]})
rows.sort(key=lambda x: x["mean_ssim"])

out = "assets/eval_low_ssim.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

import statistics
ms = [r["mean_ssim"] for r in rows]
print(f"\n  mean_ssim 分布: p1={statistics.quantiles(ms,n=100)[0]:.3f} "
      f"p5={statistics.quantiles(ms,n=100)[4]:.3f} "
      f"p25={statistics.quantiles(ms,n=4)[0]:.3f} "
      f"中位={statistics.median(ms):.3f}")

print(f"\n  === mean_ssim 最低的 40 条（跨所有 ckpt 都差）===")
print(f"  {'mean':<8}{'min':<8}{'max':<8}{'n':<4}{'char':<5}{'script':<5}img_id")
for r in rows[:40]:
    print(f"  {r['mean_ssim']:<8}{r['min_ssim']:<8}{r['max_ssim']:<8}"
          f"{r['n_ckpt']:<4}{r['char']:<5}{r['script']:<5}{r['img_id']}")

# 分组：按字符看是否有系统性问题
from collections import Counter  # noqa: E402
low = [r for r in rows if r["mean_ssim"] < 0.45]
print(f"\n  mean_ssim < 0.45 的: {len(low)}/{len(rows)}")
c = Counter(r["char"] for r in low)
dup = {k: v for k, v in c.items() if v > 1}
print(f"  其中重复出现的字符: {len(dup)} 个")
for k, v in sorted(dup.items(), key=lambda x: -x[1])[:20]:
    print(f"    {k}: {v} 次")

print(f"\n  -> {out}")
