"""绝对修正 eval200: 条件(std)一律按 csv 的 `character` 用字体重渲染, 并做双重校验。

为什么: 现有 data/top10_style23/std/*.png 的字形来源不明(繁/简/异体可能不符), 元数据
        无法判定 -> 与其猜, 不如**按每条自己的 character 重渲染**。
判据 (32×32 墨密度相关, 只在同一尺度上比, 不受细线/笔宽影响):
  corr(渲染字, 真迹图)  —— 同字应显著高于不同字; 低于阈值 = 标注本身有问题 -> 剔除。
产物:
  exp-std/data/std_fixed_eval200/*.png        修正后的 std 条件图 (256², 白底黑字)
  exp-std/data/shards_std_w7_fixed_eval200/    对应的 7px 条件 latent (与 shards_std_w7 同构)
  exp-std/csv/eval200_fixed.csv                修正后的 eval 清单(含 fix 标记; dropped 不入表)
  _ot_scratch/eval200_fix_montage.png          可视证据 (GT | 旧std | 新std | 渲染)
纯 CPU(训练占卡时也能跑)。
"""
import csv
import os
import re
import sys

import numpy as np
import torch as th
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

EV = "exp-std/csv/eval200.csv"
OUT_PNG = "exp-std/data/std_fixed_eval200"
OUT_SHARD = "exp-std/data/shards_std_w7_fixed_eval200"
OUT_CSV = "exp-std/csv/eval200_fixed.csv"
MONT = "_ot_scratch/eval200_fix_montage.png"
THR = 0.15          # corr(渲染字, 真迹) 低于此 -> 认为标注字不对, 剔除
N_SHOW = 8

# 字体: 只用**能区分繁简**的 (2026-10-03 实测: STXINGKA 缺繁体字形, 会渲染成空白 ✗)。
# 楷/行/隶 统一用 STKAITI(实测 縣/县 差异 60.5 ✓ 能区分); STSONG 作备选。
FONTS = {}
for s in (0, 3, 4):
    for p in ("_fonts/STKAITI.TTF", "_fonts/STSONG.TTF", "_fonts/Deng.ttf"):
        if os.path.exists(p):
            FONTS[s] = p
            break

os.makedirs(OUT_PNG, exist_ok=True)
os.makedirs(OUT_SHARD, exist_ok=True)
os.makedirs("_ot_scratch", exist_ok=True)

# ── 渲染: **用仓库自己的 render_glyph** (与数据集构图一致; 自搓版归一化不对, 已废弃) ──
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
from build_std_glyph_latents import render_glyph  # noqa: E402


def render_safe(ch, font_path, size=256, box_frac=0.80):
    """包一层: 返回 None 表示该字体缺字(渲染为空) -> 换字体/丢弃。"""
    if not font_path:
        return None
    try:
        arr = render_glyph(ch, font_path, size, box_frac)
    except Exception:                                        # noqa: BLE001
        return None
    if arr is None or (np.asarray(arr) < 128).mean() < 0.005:
        return None                                          # 空白 = 字体缺字形
    return arr


def ink32(arr):
    g = np.asarray(Image.fromarray(arr).resize((32, 32), Image.LANCZOS), np.float32) / 255.0
    v = 1.0 - g
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def corr(a, b):
    return float(np.sum(ink32(a) * ink32(b)))


# ── VAE (CPU) 把 7px 像素图编码成条件 latent ─────────────────────────────
from src.eval.in_mem_eval import _get_vae  # noqa: E402

dev = th.device("cpu")
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
SF = 0.18215


def to_w7_latent(png):
    """渲染图 -> 二值 -> 骨架化(1px) -> 膨胀到 ~7px -> VAE encode -> (4,32,32) f16。"""
    b = (png < 128)
    sk = skeletonize(b)
    for _ in range(3):                       # (7-1)//2 = 3 次
        sk = binary_dilation(sk)
    img = np.where(sk, 0.0, 1.0).astype(np.float32)          # 黑线白底 [-1,1] 映射
    x = th.from_numpy(img * 2 - 1)[None, None].repeat(1, 3, 1, 1).to(dev)
    with th.no_grad():
        z = vae.encode(x).latent_dist.mode() * SF
    return z[0].to(th.float16).numpy()


rows = list(csv.DictReader(open(EV, encoding="utf-8")))
print(f"[in] {EV} n={len(rows)}  阈值 corr>={THR}")

# ★ 候选字形表: 同一 glyph_id 在**训练集**里出现过的所有 character。
#   作用: 标注的 character 本身可能是简写(如 img 21968 标注 '复' 而真迹是 '復'),
#        忠实渲染错标注 = 没修。所以候选 = {本行 character} ∪ {同 glyph_id 的其它字形},
#        渲染后与真迹比对, 选最像的那个 (不需要 OCR)。
gid2chars = {}
for _r in csv.DictReader(open("exp-std/csv/train.csv", encoding="utf-8")):
    gid2chars.setdefault(str(_r["glyph_id"]), set()).add(_r["character"])
multi = sum(1 for v in gid2chars.values() if len(v) > 1)
print(f"[候选表] glyph_id 组数={len(gid2chars)}  多字形(繁/简/异体)组={multi}")


CAND_FONTS = ("_fonts/STKAITI.TTF", "_fonts/STSONG.TTF", "_fonts/Deng.ttf")


def _render_any(ch):
    """渲染字形: 逐个字体试, 返回第一个非空结果 (缺字的字体渲染为空白会自动跳过)。"""
    for p in CAND_FONTS:
        im = render_safe(ch, p)
        if im is not None:
            return im
    return None

fixed, kept_rows, dropped, corrs = [], [], [], []
char_fix = []                # (bid, 原 character, 实际渲染用的字形)
mont = []
lats, lids = [], []          # 收成一个 shard_00000.npz (条件目录必须与 shards_std_w7 同构)
for r in rows:
    bid = os.path.basename(r["image_path"])
    gt = np.asarray(Image.open(os.path.join("data/top10_style23/imgs", bid))
                    .convert("L").resize((256, 256)), np.uint8)
    old = np.asarray(Image.open(os.path.join("data/top10_style23/std", bid))
                     .convert("L").resize((256, 256)), np.uint8)
    ch = r["character"]
    # ★ 候选集: 本行 character + 同 glyph_id 在训练集里出现过的其它字形
    cands = [ch] + sorted(gid2chars.get(str(r["glyph_id"]), set()) - {ch})
    best = None
    for _cn in cands:
        _im = _render_any(_cn)
        if _im is None:
            continue
        _cc = corr(_im, gt)
        if best is None or _cc > best[1]:
            best = (_cn, _cc, _im)
    if best is None:
        dropped.append((bid, "render_fail"))
        continue
    ch_used, c_new, new = best
    c_old = corr(old, gt)
    corrs.append((c_new, c_old, bid))
    if max(c_new, c_old) < THR:
        dropped.append((bid, f"corr_low({c_new:.2f}/{c_old:.2f})"))
        continue
    use_new = c_new >= c_old
    std_png = new if use_new else old
    lats.append(to_w7_latent(std_png))
    lids.append(int(r["img_id"]))
    Image.fromarray(std_png).save(os.path.join(OUT_PNG, bid))
    nr = dict(r)
    nr["std_path"] = os.path.join(OUT_PNG, bid)
    nr["fix"] = "rendered" if use_new else "kept_old"
    nr["char_used"] = ch_used          # ← 实际用来渲染的字形(可能与 character 不同)
    kept_rows.append(nr)
    if ch_used != ch:
        char_fix.append((bid, ch, ch_used))
    if any(t in (ch or "") for t in "复復") or ch_used != ch:
        mont.insert(0, (9.9, gt, old, std_png, new))    # 这类优先看
    if use_new:
        fixed.append(bid)
    mont.append((c_new - c_old, gt, old, std_png, new))    # 差距最大的最该看

np.savez_compressed(os.path.join(OUT_SHARD, "shard_00000.npz"),
                    latents=np.stack(lats).astype(np.float16),
                    img_ids=np.array(lids))
print(f"[shard] {OUT_SHARD}/shard_00000.npz  latents={np.stack(lats).shape}")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
    w.writeheader()
    w.writerows(kept_rows)

arr = np.array([[a, b] for a, b, _ in corrs])
print(f"[结果] 保留 {len(kept_rows)}/{len(rows)} | 其中**换成重渲染 std** = {len(fixed)} | "
      f"沿用旧 std = {len(kept_rows) - len(fixed)} | 剔除 = {len(dropped)}")
print(f"[字形校正] 实际渲染字形与标注 character **不同** 的 = {len(char_fix)} 条")
for b, c0, c1 in char_fix[:12]:
    print(f"         {b}: 标注 '{c0}' -> 实际按 '{c1}' 渲染")
print(f"[相关] 渲染字 corr 中位={np.median(arr[:,0]):.3f} p10={np.percentile(arr[:,0],10):.3f} | "
      f"旧 std corr 中位={np.median(arr[:,1]):.3f} p10={np.percentile(arr[:,1],10):.3f}")
if dropped:
    print("[剔除] " + ", ".join(f"{b}({w})" for b, w in dropped[:10]))
print(f"[产物] {OUT_CSV} / {OUT_PNG}/ / {OUT_SHARD}/")

mont.sort(key=lambda x: -x[0])
mont = mont[:N_SHOW]
S = 140
canvas = Image.new("L", (S * 4, S * len(mont)), 255)
for j, (_gap, gt, old, newstd, rend) in enumerate(mont):
    for k, a in enumerate((gt, old, newstd, rend)):
        canvas.paste(Image.fromarray(a).resize((S, S)), (k * S, j * S))
canvas.save(MONT)
print(f"[图] {MONT}  每行: 目标GT | 旧std | 新std | 渲染")
