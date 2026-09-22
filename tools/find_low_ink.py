"""找「所有 ckpt 都差」的字 —— 用新的墨迹指标。

## 为什么用 ink_ssim / ink_iou / skel_iou
全图 SSIM 被 ~90% 白底抬高（「升」vs「陞」全图 0.5745 但墨迹框 0.3161），
所以要用墨迹域指标才能反映"字写得对不对"。

## 判定
对每条样本，聚合 8 个 ckpt 的:
  ink_ssim_mean / ink_iou_mean / skel_iou_mean
按 skel_iou_mean 升序（最能反映"字对不对"）找最差的。
"""
import csv
import glob
import os
import statistics
from collections import defaultdict

os.chdir("/root/Workspace/xy/DiT")

files = sorted(glob.glob("assets/ink_eval_raw/*__strict.csv"))
print(f"  {len(files)} 个 strict 逐样本文件")

acc = defaultdict(lambda: defaultdict(list))
meta = {}
for f in files:
    for r in csv.DictReader(open(f, encoding="utf-8")):
        k = r["img_id"]
        for m in ("ssim", "ink_ssim", "ink_iou", "skel_iou", "lpips", "mse"):
            v = r.get(m, "")
            if v not in ("", None):
                try:
                    acc[k][m].append(float(v))
                except ValueError:
                    pass
        if k not in meta:
            meta[k] = {"char": r.get("char", ""), "script": r.get("script", ""),
                       "idx": r.get("idx", "")}

print(f"  聚合到 {len(acc)} 个唯一样本")

rows = []
for k, d in acc.items():
    if len(d.get("ink_ssim", [])) < 3:
        continue
    rows.append({
        "img_id": k, "n": len(d["skel_iou"]),
        "skel": round(statistics.mean(d["skel_iou"]), 4),
        "ink": round(statistics.mean(d["ink_ssim"]), 4),
        "iou": round(statistics.mean(d["ink_iou"]), 4),
        "ssim": round(statistics.mean(d["ssim"]), 4),
        "lpips": round(statistics.mean(d["lpips"]), 4) if d.get("lpips") else "",
        "char": meta[k]["char"], "script": meta[k]["script"],
        "idx": meta[k]["idx"]})
rows.sort(key=lambda x: x["ink"])

print(f"\n  指标分布（{len(rows)} 条）:")
for m in ("skel", "ink", "iou", "ssim"):
    vs = [r[m] for r in rows]
    q = statistics.quantiles(vs, n=100)
    print(f"    {m:<6} p5={q[4]:.3f}  p25={q[24]:.3f}  "
          f"中位={statistics.median(vs):.3f}  p75={q[74]:.3f}")

print(f"\n  === 墨迹 SSIM 最低的 40 条（跨 8 个 ckpt 都差）===")
print(f"  {'ink':<8}{'skel':<8}{'iou':<8}{'ssim':<8}{'char':<5}{'script':<5}img_id")
for r in rows[:40]:
    print(f"  {r['skel']:<8}{r['ink']:<8}{r['iou']:<8}{r['ssim']:<8}"
          f"{r['char']:<5}{r['script']:<5}{r['img_id']}")

out = "assets/eval_low_ink.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\n  -> {out}")

# 按字聚合：哪些"字"普遍差
bychar = defaultdict(list)
for r in rows:
    bychar[r["char"]].append(r["ink"])
multi = {c: (statistics.mean(v), len(v)) for c, v in bychar.items()
         if len(v) >= 1}
worst = sorted(multi.items(), key=lambda x: x[1][0])[:30]
print(f"\n  === 按字聚合，墨迹 SSIM 最低的 30 个字 ===")
for c, (m, n) in worst:
    print(f"    {c}: ink_ssim={m:.3f}  (n={n})")
