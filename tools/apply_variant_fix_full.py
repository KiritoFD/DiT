#!/usr/bin/env python
"""一键执行「把异体字改成通行字」：csv + std g + shards 三处同步。

## 用户意图
书法数据里 GT 写的是**异体字**（刋/吿/徳/歳...），csv 标的是**异体字本身**。
改成「通行字」（刊/告/德/岁...）后：
  - csv 的 character        = 通行字
  - std g（从 character 渲染）= 通行字的规整骨架
  - GT（书法）              = 异体字
-> 模型学会「从通行字骨架写出书家的异体写法」—— 训练集多些通行字是好事。

## 三处同步
1. assets/train_50k_v2_fixed.csv         character 替换
2. data/50k/std/{id}.png                 用 STKAITI 重渲染
3. data/50k/shards_std_fixed/*.npz       对应 latent 替换（不重编全量）

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/apply_variant_fix_full.py --min-conf 0.9
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--review", default="assets/stage2_review.csv")
ap.add_argument("--train", default="assets/train_50k_v2_fixed.csv")
ap.add_argument("--min-conf", type=float, default=0.9)
ap.add_argument("--font", default="_fonts/STKAITI.TTF")
ap.add_argument("--shards", default="data/50k/shards_std_fixed")
ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--dry-run", type=int, default=1)
a = ap.parse_args()

# ── 1) 收集要改的 ────────────────────────────────────────────────────
rev = list(csv.DictReader(open(a.review, encoding="utf-8")))
todo = []
for r in rev:
    if r["verdict"] != "strong":
        continue
    try:
        conf = float(r["s2_conf"] or 0)
    except ValueError:
        continue
    if conf < a.min_conf:
        continue
    old, new = r["char"], r["s2_pred"]
    if not new or new in ("?", "") or old == new:
        continue
    todo.append((int(r["idx"]), r["img_id"], old, new))

print(f"  strong+conf>={a.min_conf}: {len(todo)} 条")
from collections import Counter  # noqa: E402
pairs = Counter((o, n) for _, _, o, n in todo)
print(f"  (old->new) 组合: {len(pairs)} 种")

# 标记可疑映射
#  ① 简繁层级相同（t2s 结果一样）-> 可能是异体/误认，但也可能对（如 爲/為 都是「为」的异体）
#  ② ★ 反向：old 能被 t2s 简化（说明 old 是繁体/通行），而 new 不能
#     -> 说明把"通行字"改成了"更生僻的字" ✗
#     例：隱(t2s->隐) vs 隠(t2s->隠) —— 这个映射是反的
from opencc import OpenCC  # noqa: E402
_t2s = OpenCC("t2s")
suspect = []
reverse = []
for (o, n), c in pairs.items():
    if _t2s.convert(n) == _t2s.convert(o) and n != o:
        suspect.append((o, n, c))
    # 反向检测
    o_simplifiable = _t2s.convert(o) != o
    n_simplifiable = _t2s.convert(n) != n
    if o_simplifiable and not n_simplifiable:
        reverse.append((o, n, c))
print(f"  ⚠ 简繁层级相同（可能误认）: {len(suspect)} 种")
for o, n, c in sorted(suspect, key=lambda x: -x[2])[:10]:
    print(f"      {o} -> {n}  ×{c}   (t2s: {_t2s.convert(o)} / {_t2s.convert(n)})")
print(f"  ⚠★ 反向（把通行字改成生僻字）: {len(reverse)} 种")
for o, n, c in sorted(reverse, key=lambda x: -x[2])[:10]:
    print(f"      {o} -> {n}  ×{c}   (t2s: {_t2s.convert(o)} / {_t2s.convert(n)})")
if reverse:
    print("      -> 这些将从待改列表中剔除")
rev_set = {(o, n) for o, n, _ in reverse}

if a.dry_run:
    print("\n  [dry-run] 去掉 --dry-run 0 执行")
    raise SystemExit(0)

# 剔除反向映射
if rev_set:
    before = len(todo)
    todo = [t for t in todo if (t[2], t[3]) not in rev_set]
    print(f"  剔除反向映射: {before} -> {len(todo)} 条")

# ── 2) 改 train csv ─────────────────────────────────────────────────
rows = list(csv.DictReader(open(a.train, encoding="utf-8")))
cols = list(rows[0].keys())
m = {i: (o, n) for i, _, o, n in todo}
n_chg = 0
for i, r in enumerate(rows):
    if i in m and r["character"] == m[i][0]:
        r["character"] = m[i][1]
        n_chg += 1
with open(a.train, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"\n  ✓ csv: 改了 {n_chg} 行")

# ── 3) 重渲染 std g ─────────────────────────────────────────────────
from skimage.morphology import skeletonize  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402


def render_skeleton(ch, font_path, size):
    font = ImageFont.truetype(font_path, int(size * 0.82))
    canvas = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(canvas)
    bb = d.textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), ch,
           font=font, fill=0)
    ink = np.asarray(canvas) < 127
    sk = binary_dilation(skeletonize(ink), iterations=1)
    ys, xs = np.where(sk)
    if len(ys) == 0:
        return None
    crop = sk[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = crop.shape
    out = np.full((size, size), 255, np.uint8)
    oy, ox = (size - hh) // 2, (size - ww) // 2
    out[oy:oy + hh, ox:ox + ww] = np.where(crop, 0, 255)
    return Image.fromarray(out)


n_rend = n_fail = 0
new_imgs = {}
for i, iid, old, new in todo:
    p = f"data/50k/std/{int(iid):06d}.png"
    img = render_skeleton(new, a.font, a.size)
    if img is None:
        n_fail += 1
        continue
    img.save(p)
    new_imgs[int(iid)] = p
    n_rend += 1
print(f"  ✓ std g: 重渲染 {n_rend} 张，失败 {n_fail}")

# ── 4) 重建 shards 对应行 ───────────────────────────────────────────
from diffusers.models import AutoencoderKL  # noqa: E402
from torchvision import transforms  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
tf = transforms.Compose([transforms.Resize((a.size, a.size)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])

files = sorted(glob.glob(os.path.join(a.shards, "*.npz")))
n_shard = 0
for f in files:
    with np.load(f) as d:
        ids = d["img_ids"].copy()
        lats = d["latents"].copy()
    hit = [k for k, v in enumerate(ids) if int(v) in new_imgs]
    if not hit:
        continue
    for k in hit:
        im = Image.open(new_imgs[int(ids[k])]).convert("RGB")
        if im.size != (a.size, a.size):
            im = im.resize((a.size, a.size), Image.LANCZOS)
        x = tf(im)[None].to(dev)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * 0.18215
        lats[k] = lat.cpu().float().numpy().astype(np.float16)[0]
        n_shard += 1
    np.savez_compressed(f, latents=lats, img_ids=ids)
print(f"  ✓ shards: 替换 {n_shard} 条（涉及 {len(files)} 个 shard 文件）")

with open("assets/variant_fixed_ids.csv", "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "img_id", "old_char", "new_char"])
    for i, iid, o, n in todo:
        w.writerow([i, iid, o, n])
print(f"  ✓ -> assets/variant_fixed_ids.csv")
print(f"\n  完成: csv {n_chg} / std {n_rend} / shards {n_shard}")
