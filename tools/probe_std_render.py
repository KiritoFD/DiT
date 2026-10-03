"""查明: 训练集 std PNG 是哪个字体/渲染器生成的; 以及 '复' 在该字体下是否渲染成 '復'。

背景: eval200 第 8 行 img_id=21968 标注 character=复, 但真迹是 復。
如果训练集 std 用的字体把 U+590D(复) 画成 traditional 字形, 那 eval 条件图必须用同一字体重渲染;
而不是按 character 字段换个"更对"的字。这个脚本给出决定性答案。
"""
import csv
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")

S = 128


def load(p):
    if not p or not os.path.exists(p):
        return None
    im = Image.open(p).convert("L").resize((S, S))
    a = 255.0 - np.asarray(im, dtype=np.float32)
    return a / (a.std() + 1e-6)


def corr(a, b):
    if a is None or b is None:
        return float("nan")
    return float((a * b).mean())


# ---- 1. 训练集里含 复/復 的行 ----
tr = list(csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8")))
fu = [r for r in tr if "复" in r["character"] or "復" in r["character"]]
print(f"[train] 26k 中含 复/復 的行 = {len(fu)}")
for r in fu[:8]:
    print(f"    img_id={r['img_id']:>6} char={r['character']} glyph={r['glyph_id']} "
          f"script={r['script']} source={r['source']}")

# ---- 2. 字体清单 ----
fonts = []
for pat in ("_fonts/*.ttf", "_fonts/*.otf", "_fonts/*.ttc",
            "/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf",
            "/usr/share/fonts/**/*.ttc"):
    fonts += glob.glob(pat, recursive=True)
fonts = sorted(set(fonts))
print(f"\n[fonts] 共 {len(fonts)} 个")
for f in fonts:
    print("    ", f)


def render(ch, fp, size=100):
    try:
        font = ImageFont.truetype(fp, size)
    except Exception:
        return None
    im = Image.new("L", (S, S), 255)
    ImageDraw.Draw(im).text((S // 2, S // 2), ch, font=font, fill=0, anchor="mm")
    a = 255.0 - np.asarray(im, dtype=np.float32)
    if a.max() < 1:
        return None
    return a / (a.std() + 1e-6)


# ---- 3. eval 21968 的三方对照 ----
gt = load("data/top10_style23/imgs/021968.png")
std_ev = load("data/top10_style23/std/021968.png")
print(f"\n[eval 21968] GT墨量={None if gt is None else round(float(gt.mean()), 3)}  "
      f"缓存std墨量={None if std_ev is None else round(float(std_ev.mean()), 3)}  "
      f"corr(GT,缓存std)={corr(gt, std_ev):+.4f}")

# 训练侧同字对照: 优先级 = 同 character 且同 source 家族
tgt = next((r for r in fu if r["character"] == "復"
            and r["source"] == "hcsu_bei"), None) or (fu[0] if fu else None)
tr_std = tr_gt = None
if tgt:
    tr_std = load(os.path.join("data/top10_style23/std",
                               os.path.basename(tgt["image_path"])))
    tr_gt = load(os.path.join("data/top10_style23/imgs",
                              os.path.basename(tgt["image_path"])))
    if tr_std is None:
        tr_std = load(tgt.get("std_path"))
    print(f"[train 参照] img_id={tgt['img_id']} char={tgt['character']} "
          f"glyph={tgt['glyph_id']}  script={tgt['script']}  source={tgt['source']}")
    print(f"            corr(训练std, 训练GT) = {corr(tr_std, tr_gt):+.4f}   "
          f"corr(训练std, eval缓存std) = {corr(tr_std, std_ev):+.4f}")

# ---- 4. 逐字体渲染 复 / 復, 看谁最像 ----
print("\n[字体判定]  corr(渲染, 训练std) | corr(渲染, eval缓存std) | corr(渲染, eval GT)")
rows = []
for fp in fonts:
    a1, a2 = render("复", fp), render("復", fp)
    v1 = (corr(a1, tr_std), corr(a1, std_ev), corr(a1, gt))
    v2 = (corr(a2, tr_std), corr(a2, std_ev), corr(a2, gt))
    rows.append((os.path.basename(fp), v1, v2))
    print(f"  {os.path.basename(fp):<30} 复 {v1[0]:+.3f}/{v1[1]:+.3f}/{v1[2]:+.3f}    "
          f"復 {v2[0]:+.3f}/{v2[1]:+.3f}/{v2[2]:+.3f}")

if tr_std is not None:
    bs = max(rows, key=lambda t: max(t[1][0], t[2][0]))
    print(f"\n>>> 最像**训练集 std** 的是字体 {bs[0]}, 且该字体下 "
          f"复={max(bs[1][0], bs[2][0]):+.3f} vs "
          f"復={bs[2][0]:+.3f} -> 训练集用的是 "
          f"{'復' if bs[2][0] > bs[1][0] else '复'} 字形")
