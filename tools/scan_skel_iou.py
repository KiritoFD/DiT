"""骨架 IoU 扫描：找出「csv 的 character」与「GT 实际字形」不一致的样本。

## 原理
`std g`（data/50k/std/{id}.png）是从 csv 的 `character` 用固定字体渲染的**骨架**。
把 GT 图（data/50k/imgs/{id}.png）也骨架化，两者对齐后算 IoU：
  - IoU 高 -> csv 的 character 与 GT 一致 ✓
  - IoU 低 -> 错配 ✗（如 csv=升 但 GT 写「陞」）

## 为什么不用 OCR
实测 rapidocr/VLM 对「陞」这类异体字都认错（输出「隆」「壁」），
字表不覆盖，路线不可行。

## 关键：对齐
GT 是书法（有风格偏移、笔画粗细差异），std 是规整骨架。
必须先做**包围盒归一化**（裁到墨迹 bbox 再缩放到同尺寸），否则 IoU 会被
留白和位置差异主导（历史实测未归一化时 IoU 只有 0.01-0.05）。

用法:
  python tools/scan_skel_iou.py --csv assets/train_50k_v2_fixed.csv \
      --out assets/skel_iou_train.csv --limit 2000
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--thr", type=float, default=0.20, help="低于此 IoU 视为可疑")
a = ap.parse_args()


def to_skel_mask(im, size=128, binarize_thr=127):
    """图 -> 二值墨迹 -> 裁 bbox -> 等比缩放到 size -> 返回 bool mask。"""
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    ink = g < binarize_thr                       # 墨迹为 True
    if ink.sum() < 10:
        return np.zeros((size, size), bool)
    ys, xs = np.where(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]
    h, w = crop.shape
    s = max(h, w)
    # 等比缩放到 (size, size)，保持长宽比 + 居中
    sc = size / s
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    img = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (nw, nh), Image.NEAREST)
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(img) > 127
    return canvas.astype(bool)


def skeletonize(mask):
    """简单细化：反复腐蚀-条件判断太重，这里用「形态学骨架近似」——
    对 IoU 比较够用（两边都做同样处理，偏差抵消）。"""
    try:
        from skimage.morphology import skeletonize as _sk
        return _sk(mask)
    except ImportError:
        return mask


def iou(a, b, dilate=2):
    """带轻微膨胀的 IoU（容忍书法与骨架的线宽差异）。"""
    if dilate:
        a = binary_dilation(a, iterations=dilate)
        b = binary_dilation(b, iterations=dilate)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / max(union, 1)


rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
print(f"  {a.csv}: {len(rows)} 条  (size={a.size})", flush=True)

out = []
miss = 0
for i, r in enumerate(rows):
    iid = r.get("old_50k_id", "").strip()
    if not iid:
        continue
    gt_p = f"data/50k/imgs/{int(iid):06d}.png"
    std_p = r.get("std_path", "").strip()
    if not std_p:
        std_p = f"data/50k/std/{int(iid):06d}.png"
    if not os.path.exists(gt_p) or not os.path.exists(std_p):
        miss += 1
        continue
    try:
        mg = skeletonize(to_skel_mask(Image.open(gt_p), a.size))
        ms = skeletonize(to_skel_mask(Image.open(std_p), a.size))
        v = iou(mg, ms)
    except Exception:
        miss += 1
        continue
    out.append({"idx": i, "id": iid, "character": r.get("character", ""),
                "callig": r.get("calligrapher", ""),
                "script": r.get("script", ""),
                "src": r.get("src_image_path", ""),
                "iou": round(v, 4)})
    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{len(rows)}", flush=True)

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

v = np.array([x["iou"] for x in out])
bad = [x for x in out if x["iou"] < a.thr]
print(f"\n  完成 {len(out)} 条（缺文件 {miss}）")
print(f"  IoU 分布: p1={np.percentile(v,1):.3f} p5={np.percentile(v,5):.3f} "
      f"p25={np.percentile(v,25):.3f} p50={np.median(v):.3f} "
      f"p75={np.percentile(v,75):.3f} p95={np.percentile(v,95):.3f}")
print(f"  IoU < {a.thr}: {len(bad)} 条 ({len(bad)/max(len(out),1)*100:.1f}%)")
print(f"\n  === 最可疑的 30 条 ===")
for x in sorted(out, key=lambda z: z["iou"])[:30]:
    print(f"    iou={x['iou']:.3f}  {x['character']}  "
          f"({x['callig']}/{x['script']})  id={x['id']}")
print(f"\n  -> {a.out}")
