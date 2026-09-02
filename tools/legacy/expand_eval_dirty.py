#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1) 扩展 eval 集到 ~500 张 (从 eval.csv 472 行)
2) 统计全量 train 数据中"不干净"图片的比例 (大面积灰色涂抹, 非黑白)
"""
import os, sys, csv, json, time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/root/Workspace/xy/DiT"

def fix_path(p):
    if p.startswith("final_images/"):
        return p.replace("final_images/", "final_imgs_256/", 1)
    return p

# ── 1) 扩展 eval 集 ──
print("=" * 60)
print("=== 1. 扩展 eval 集 ===")
eval_csv = os.path.join(BASE, "5script", "eval.csv")
rows = list(csv.DictReader(open(eval_csv, encoding="utf-8")))
print(f"eval.csv: {len(rows)} rows")

# 修路径 + 检查存在
valid = []
for r in rows:
    p = fix_path(r["image_path"])
    full = os.path.join(BASE, p)
    if os.path.isfile(full):
        r["image_path"] = p
        valid.append(r)
print(f"valid (exist on disk): {len(valid)}")

# 写扩展 eval
out_path = os.path.join(BASE, "5script", "eval500_clean.csv")
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=valid[0].keys())
    w.writeheader()
    for r in valid:
        w.writerow(r)
print(f"wrote: {out_path} ({len(valid)} rows)")

from collections import Counter
sc = Counter(r["script"] for r in valid)
print(f"script 分布: {dict(sc)}")

# ── 2) 不干净图片统计 ──
print("\n" + "=" * 60)
print("=== 2. 不干净图片统计 (全量 train) ===")
train_csv = os.path.join(BASE, "5script", "train_top30_clean.csv")
train_rows = list(csv.DictReader(open(train_csv, encoding="utf-8")))
print(f"train: {len(train_rows)} rows")

# 采样: 先分析 eval500 (快速), 再采样 train
all_paths = []
for r in train_rows:
    p = fix_path(r["image_path"])
    full = os.path.join(BASE, p)
    if os.path.isfile(full):
        all_paths.append(full)
print(f"train images found: {len(all_paths)}")

# 分析单张图: 返回 (grey_ratio, mean_val, std_val)
def analyze_image(p):
    try:
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        grey = ((arr > 0.15) & (arr < 0.85)).mean()  # 灰色像素比例
        return (float(grey), float(arr.mean()), float(arr.std()))
    except Exception:
        return None

# 先对 eval500 分析
print(f"\n--- eval500 ({len(valid)} 张) ---")
t0 = time.time()
eval_results = []
with ThreadPoolExecutor(max_workers=32) as pool:
    eval_paths = [os.path.join(BASE, r["image_path"]) for r in valid]
    eval_results = list(pool.map(analyze_image, eval_paths))
eval_results = [r for r in eval_results if r is not None]
t_eval = time.time() - t0
print(f"analyzed {len(eval_results)} in {t_eval:.1f}s")

# 统计
grey_ratios = [r[0] for r in eval_results]
grey_ratios = np.array(grey_ratios)
for thresh in [0.1, 0.2, 0.3, 0.5]:
    n = (grey_ratios > thresh).sum()
    print(f"  grey_ratio > {thresh}: {n}/{len(grey_ratios)} ({100*n/len(grey_ratios):.1f}%)")
print(f"  grey_ratio: mean={grey_ratios.mean():.4f} median={np.median(grey_ratios):.4f} "
      f"p95={np.percentile(grey_ratios, 95):.4f} max={grey_ratios.max():.4f}")

# 最脏的 10 张
dirty_idx = np.argsort(grey_ratios)[::-1][:10]
print(f"\n  最脏的 10 张 (eval500):")
for idx in dirty_idx:
    r = valid[idx]
    print(f"    {r['image_path']} grey={grey_ratios[idx]:.3f} char={r['character']} script={r['script']}")

# 全量 train 分析 (多线程, 分批报告)
print(f"\n--- train ({len(all_paths)} 张) ---")
t0 = time.time()
train_results = []
BATCH = 5000
for i in range(0, len(all_paths), BATCH):
    batch = all_paths[i:i+BATCH]
    with ThreadPoolExecutor(max_workers=32) as pool:
        batch_res = list(pool.map(analyze_image, batch))
    train_results.extend([r for r in batch_res if r is not None])
    elapsed = time.time() - t0
    done = len(train_results)
    rate = done / elapsed
    eta = (len(all_paths) - done) / rate
    print(f"  {done}/{len(all_paths)} ({rate:.0f} img/s, ETA {eta/60:.0f}min, "
          f"grey>0.3: {sum(1 for r in train_results if r[0]>0.3)})", flush=True)

t_train = time.time() - t0
print(f"\n  train analyzed {len(train_results)} in {t_train:.0f}s ({len(train_results)/t_train:.0f} img/s)")

train_grey = np.array([r[0] for r in train_results])
print(f"\n=== 不干净图片统计 (train {len(train_results)} 张) ===")
for thresh in [0.05, 0.1, 0.2, 0.3, 0.5]:
    n = (train_grey > thresh).sum()
    print(f"  grey_ratio > {thresh}: {n}/{len(train_grey)} ({100*n/len(train_grey):.2f}%)")
print(f"  grey_ratio: mean={train_grey.mean():.4f} median={np.median(train_grey):.4f} "
      f"p95={np.percentile(train_grey, 95):.4f} max={train_grey.max():.4f}")

# 按 script 分组统计
print(f"\n=== 按 script 分组 ===")
script_grey = {}
for r, g in zip(train_rows[:len(train_results)], train_grey):
    s = r["script"]
    script_grey.setdefault(s, []).append(g)
for s in sorted(script_grey.keys()):
    gs = np.array(script_grey[s])
    dirty = (gs > 0.3).sum()
    print(f"  {s}: {len(gs)} 张, grey>0.3: {dirty} ({100*dirty/len(gs):.2f}%), "
          f"mean_grey={gs.mean():.4f}")

# 保存结果
summary = {
    "eval500": {
        "n": len(eval_results),
        "grey_ratio_mean": float(grey_ratios.mean()),
        "grey_ratio_p95": float(np.percentile(grey_ratios, 95)),
        "dirty_count": int((grey_ratios > 0.3).sum()),
    },
    "train": {
        "n": len(train_results),
        "grey_ratio_mean": float(train_grey.mean()),
        "grey_ratio_p95": float(np.percentile(train_grey, 95)),
        "dirty_count": int((train_grey > 0.3).sum()),
        "dirty_ratio": float((train_grey > 0.3).sum() / len(train_grey)),
    },
    "by_script": {
        s: {
            "n": len(script_grey[s]),
            "dirty_count": int((np.array(script_grey[s]) > 0.3).sum()),
            "dirty_ratio": float((np.array(script_grey[s]) > 0.3).sum() / len(script_grey[s])),
            "grey_mean": float(np.mean(script_grey[s])),
        }
        for s in sorted(script_grey.keys())
    },
}
out = os.path.join(BASE, "tools", "dirty_image_stats.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n保存: {out}")
