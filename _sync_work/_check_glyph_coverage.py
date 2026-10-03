# -*- coding: utf-8 -*-
"""_check_glyph_coverage.py — 标准字(g)覆盖与简繁/异体字体检.

背景: 训练数据是书法古籍 (fame3/tongji/UniCalli), 字符大量是**繁体/异体字**
      (如 㑺), 而 render() 用的字库是 simkai/STKAITI/NotoSerifSC 等。
      若字库缺字形, PIL 会画 .notdef (豆腐块 □), 像素数很多 -> 能骗过
      `(a<250).sum() < 10` 的"渲染成功"判据 -> std 骨架变成"方框", 而不是 miss。
目标:
  1) 数据字符集里有多少能被 simkai / STKAITI 正常渲染 (不是豆腐块)
  2) 现有 std_skel PNG 里有多少是可疑的"方框" (前景占比异常高)
  3) 训练/评测的 g 命中率 (key2uid 覆盖率)
"""
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FONTS = ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf", "simhei.ttf", "SIMLI.TTF"]

# ---------- 1) 数据字符集 ----------
rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
chars = [r["character"] for r in rows if r.get("character")]
uniq = sorted(set(chars))
print(f"[data] rows={len(rows)} unique chars={len(uniq)}")
cnt = Counter(chars)
print(f"[data] top10 高频字: {cnt.most_common(10)}")

# ---------- 2) 豆腐块检测器 ----------
# .notdef 通常是空心方框: 骨架化后是"矩形环", 前景占比高于正常汉字,
# 且 4 个角都接近图边。用"渲染 - 替换字符"比较最稳。
def render_arr(ch, fp, size=200):
    font = ImageFont.truetype(fp, size)
    img = Image.new("L", (256, 256), 255)
    ImageDraw.Draw(img).text((128, 128), ch, font=font, fill=0, anchor="mm")
    return np.asarray(img)


def is_tofu(ch, fp):
    """豆腐块判据: 与 .notdef 参考字形高度相似 (IoU>0.85)."""
    a = render_arr(ch, fp)
    ref = render_arr("\uFFFF", fp)      # 不可映射 -> .notdef
    ab, rb = (a < 200), (ref < 200)
    if rb.sum() < 10:
        return False, 0.0
    iou = (ab & rb).sum() / max((ab | rb).sum(), 1)
    return iou > 0.85, iou


print("\n[字形覆盖] 对每个字库检查 uniq 字符 (取前 3000 个抽样以控时间)")
sample = uniq[:3000]
for f in FONTS:
    fp = os.path.join("tools/fonts", f)
    if not os.path.isfile(fp):
        print(f"  [{f:20s}] 字体文件不存在")
        continue
    miss = tofu = 0
    for ch in sample:
        try:
            a = render_arr(ch, fp)
        except Exception:
            miss += 1
            continue
        if (a < 250).sum() < 10:
            miss += 1
            continue
        t, _ = is_tofu(ch, fp)
        if t:
            tofu += 1
    print(f"  [{f:20s}] 空白/异常={miss:4d}  豆腐块={tofu:4d}  "
          f"({100*(miss+tofu)/len(sample):.1f}% 不可用)")

# ---------- 3) 现有 std_skel PNG 体检 ----------
print("\n[std_skel PNG 体检] 检测异常'方框'骨架")
for d in ("data/skel/std_skel3_base_png", "data/skel/final_skel3_base"):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    if not fs:
        print(f"  [{d}] 无 PNG")
        continue
    take = fs[:2000]
    fg = []
    for p in take:
        a = np.asarray(Image.open(p).convert("L"))
        fg.append((a < 127).mean())      # 骨架像素占比
    fg = np.array(fg)
    # 方框骨架: 前景占比高且分布集中; 正常字骨架占比通常 3-12%
    susp = (fg > 0.18).sum()
    print(f"  [{d}] n={len(fg)} 前景占比 p50={np.percentile(fg,50):.3f} "
          f"p95={np.percentile(fg,95):.3f} max={fg.max():.3f} | 可疑(>0.18)={susp}")

# ---------- 4) g 命中率 (key2uid) ----------
print("\n[g 命中率]")
for f in ("data/skel/std_skel3_base_key2uid.csv", ):
    if not os.path.isfile(f):
        print(f"  {f} 不存在")
        continue
    kr = list(csv.DictReader(open(f, encoding="utf-8")))
    print(f"  {f}: {len(kr)} keys; 列={list(kr[0].keys()) if kr else '?'}")
    miss = [r for r in kr if not os.path.exists(
        os.path.join("data/skel/std_skel3_base_png", f"{r.get('uid', '')}.png"))]
    print(f"  对应 PNG 缺失: {len(miss)}")
