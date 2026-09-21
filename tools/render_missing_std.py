"""用系统字体渲染标准字形 -> 骨架化 -> 补上缺失的 std 图。

## 为什么要对比字体
现有 data/50k/std/*.png 是**骨架图**（256x256 二值，1px 细线），
来自某个中文字体渲染后骨架化。补字时必须用**同一个字体**，
否则这 20 个字的 g 分布会和其他字不一致。

做法: 用 4 个字体渲染几个**已知字**（它们在 std 里已有图），
      骨架化后和现有 std 图算 IoU，选 IoU 最高的字体。
"""
import csv
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

os.chdir("/root/Workspace/xy/DiT")

FONTS = {
    "Deng": "_fonts/Deng.ttf",
    "STKAITI": "_fonts/STKAITI.TTF",
    "STSONG": "_fonts/STSONG.TTF",
    "msyh": "_fonts/msyh.ttc",
}
SIZE = 256


def render_skeleton(ch, font_path, size=SIZE, font_px=None):
    """渲染单字 -> 二值 -> 骨架 -> 裁到包围盒并居中到 size x size。"""
    fp = font_px or int(size * 0.82)
    font = ImageFont.truetype(font_path, fp)
    canvas = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(canvas)
    # 居中绘制
    bbox = d.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), ch,
           font=font, fill=0)
    a = np.array(canvas)
    bw = a < 128
    if bw.sum() < 10:
        return None
    ys, xs = np.where(bw)
    crop = canvas.crop((int(xs.min()), int(ys.min()),
                        int(xs.max()) + 1, int(ys.max()) + 1))
    w2, h2 = crop.size
    sc = (size * 0.9) / max(w2, h2)
    crop = crop.resize((max(1, int(w2 * sc)), max(1, int(h2 * sc))),
                       Image.BILINEAR)
    c2 = Image.new("L", (size, size), 255)
    c2.paste(crop, ((size - crop.size[0]) // 2, (size - crop.size[1]) // 2))
    a2 = np.array(c2)
    return skeletonize(a2 < 128)


def to_skel_file(path):
    im = Image.open(path).convert("L")
    a = np.array(im)
    bw = a < 128
    ys, xs = np.where(bw)
    if len(ys) < 10:
        return None
    crop = im.crop((int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = crop.size
    sc = (SIZE * 0.9) / max(w, h)
    crop = crop.resize((max(1, int(w * sc)), max(1, int(h * sc))),
                       Image.BILINEAR)
    c2 = Image.new("L", (SIZE, SIZE), 255)
    c2.paste(crop, ((SIZE - crop.size[0]) // 2, (SIZE - crop.size[1]) // 2))
    return skeletonize(np.array(c2) < 128)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


# ── 1) 用已知字对比字体 ───────────────────────────────────────────────
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
std_of = {}
for r in rows:
    std_of.setdefault(r["character"], r["std_path"])

TEST_CHARS = ["並", "來", "亂", "亞", "國", "學", "書", "東", "樂", "風"]
print("  === 字体对比（用已知字算 IoU）===")
scores = {}
for fname, fpath in FONTS.items():
    if not os.path.exists(fpath):
        print(f"    {fname}: 字体文件不存在")
        continue
    vals = []
    for ch in TEST_CHARS:
        sp = std_of.get(ch)
        if not sp or not os.path.exists(sp):
            continue
        ref = to_skel_file(sp)
        got = render_skeleton(ch, fpath)
        vals.append(iou(ref, got))
    if vals:
        scores[fname] = float(np.mean(vals))
        print(f"    {fname:<10} IoU={np.mean(vals):.4f}  (n={len(vals)})")

if not scores:
    raise SystemExit("没有可用的字体对比结果")

best = max(scores, key=scores.get)
print(f"\n  ★ 最匹配的字体: {best} (IoU={scores[best]:.4f})")

# ── 2) 补缺失的 20 个字 ──────────────────────────────────────────────
wide = list(csv.DictReader(open("assets/mismatch_wide.csv", encoding="utf-8")))
miss_chars = sorted(set(r["char_true"] for r in wide
                        if r["char_true"] not in std_of))
print(f"\n  === 需要补的字: {len(miss_chars)} ===")
print(f"    {''.join(miss_chars)}")

outdir = "data/50k/std"
existing = set()
import glob
for p in glob.glob(os.path.join(outdir, "*.png")):
    existing.add(os.path.basename(p))

# 新文件名用 9xxxxx 段（避开已有编号）
start = 900000
added = {}
for k, ch in enumerate(miss_chars):
    sk = render_skeleton(ch, FONTS[best])
    if sk is None:
        print(f"    ✗ {ch}: 渲染失败")
        continue
    # 骨架 -> 256x256 二值图（白底黑线）
    img = Image.fromarray(np.where(sk, 0, 255).astype(np.uint8))
    name = f"{start + k:06d}.png"
    img.save(os.path.join(outdir, name))
    added[ch] = os.path.join(outdir, name)
    print(f"    ✓ {ch} -> {name}")

print(f"\n  补了 {len(added)} 个字")

# 存映射
import json
json.dump(added, open("assets/_std_added.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"  written assets/_std_added.json")
