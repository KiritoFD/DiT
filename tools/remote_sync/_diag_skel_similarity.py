# -*- coding: utf-8 -*-
"""楷隶子集 vs 标准字库 skel 相似度诊断(远程 CPU)。

统计:
  1. kailishu_train.csv 总样本/书家/每书家分布
  2. 按书家分层采样样本, 对比其 GT skeleton(final_skeleton) vs 标准字形骨架(std_skel)
     IoU(质心对齐), 输出总体分布 + 按书家平均相似度

用法: python _diag_skel_similarity.py [sample_each_callig]
"""
import csv, os, sys, json, random
import numpy as np
from PIL import Image
from collections import defaultdict

random.seed(0)
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 3
HERE = os.path.dirname(os.path.abspath(__file__))

rows = list(csv.DictReader(open("kailishu_train.csv", encoding="utf-8")))
print(f"总样本: {len(rows)}")
callig = defaultdict(int)
for r in rows:
    callig[int(r["calligrapher_id"])] += 1
print(f"总书家: {len(callig)}")
sizes = sorted(callig.values())
print(f"每书家样本分布: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
print(f"  >100样本书家: {sum(1 for s in sizes if s>100)} | 样本前10书家占比: {sum(sorted(sizes,reverse=True)[:10])/len(rows)*100:.1f}%")

# 按书家分组
by_callig = defaultdict(list)
for r in rows:
    by_callig[int(r["calligrapher_id"])].append(r)

def load_gt_skel(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return (a > 64)

def load_std_skel(fname):
    # std_skel/{kai|li}/U+XXXXX.png (标准字体渲染骨架, 二值)
    p = os.path.join(HERE, "std_skel", fname)
    if not os.path.exists(p):
        return None
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    return (a > 64)

def centroid(a):
    ys, xs = np.nonzero(a)
    return (ys.mean(), xs.mean()) if len(xs) else (0,0)

def align(a):
    cy, cx = centroid(a)
    from scipy.ndimage import shift
    return shift(a.astype(np.float32), shift=(128-cy, 128-cx), order=0, mode='constant', cval=0) > 0.5

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)

# 采样: 每书家取 N_EACH 样本
sampled, per_callig_iou = [], defaultdict(list)
missing_std = 0
for cid, rlist in by_callig.items():
    if len(rlist) < 1: continue
    picked = random.sample(rlist, min(N_EACH, len(rlist)))
    for r in picked:
        img_id = os.path.basename(r["image_path"])[:-4]
        script_key = {"0":"kai","4":"li"}.get(r["script_id"], "")
        if not script_key:
            continue
        # 标准字形骨架文件名
        std_fn = os.path.join(script_key, f"U+{ord(r['character']):05X}.png")
        gtsk = load_gt_skel(os.path.join("final_skeleton", f"{img_id}.png"))
        stsk = load_std_skel(std_fn)
        if stsk is None or stsk.sum() < 20:
            missing_std += 1
            continue
        gta, sta = align(gtsk), align(stsk)
        v = iou(sta, gta)
        sampled.append(v)
        per_callig_iou[cid].append(v)

all_v = np.array(sampled)
print(f"\n采样样本: {len(all_v)} (标准字缺失 {missing_std})")
print(f"总体 标准skel vs GT骨架 IoU:")
print(f"  mean={all_v.mean():.3f} median={np.median(all_v):.3f} p25={np.percentile(all_v,25):.3f} p75={np.percentile(all_v,75):.3f} min={all_v.min():.3f} max={all_v.max():.3f}")

# 按书家聚合
print(f"\n按书家平均 IoU 分布 (共 {len(per_callig_iou)} 书家):")
avg = {cid: np.mean(vs) for cid, vs in per_callig_iou.items()}
avgs = sorted(avg.values())
print(f"  median={np.median(avgs):.3f} p10={np.percentile(avgs,10):.3f} p90={np.percentile(avgs,90):.3f}")
print(f"  >0.3 的书家: {sum(1 for a in avgs if a>0.3)} | >0.2: {sum(1 for a in avgs if a>0.2)}")
# 最好的几个书家
best = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:8]
print("  最接近标准字库的 top8 书家:")
for cid, v in best:
    nm = next((r["calligrapher"] for r in by_callig[cid]), "?")
    print(f"    callig={cid} {nm}: avg IoU={v:.3f} (样本{len(by_callig[cid])})")

# 存结果
with open("skel_similarity_report.json", "w", encoding="utf-8") as f:
    json.dump({"total_rows": len(rows), "n_callig": len(callig),
               "overall": {"mean": float(all_v.mean()), "median": float(np.median(all_v))},
               "per_callig_median": float(np.median(avgs))}, f, ensure_ascii=False, indent=2)
print("\nsaved -> skel_similarity_report.json")
