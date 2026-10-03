"""把 REPA 的 **patch 特征缓存** 池化成 CLS 特征 npz, 喂给 tools/build_multistyle_k4.py。

背景: data/dino_cache/top10_v1/{feats.f16, ids.npy, meta.json} 存的是每张图的 patch 特征
      (5.1 GB / 38,583 张 ≈ 256 token × 384 维)。而 build_multistyle_k4.py 要的是
      npz(feat(N,D), calligs(N,), scripts(N,)) 这种"每图一个向量"。
      -> 这里对 patch 维做 mean-pool, 不用再跑一次 DINO 前向。

用法: python tools/dino_cache_to_cls_npz.py \
        --cache data/dino_cache/top10_v1 \
        --csv exp-std/csv/train.csv \
        --out assets/dino_feat_top10.npz --chunk 2048
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/dino_cache/top10_v1")
ap.add_argument("--csv", default="exp-std/csv/train.csv")
ap.add_argument("--out", default="assets/dino_feat_top10.npz")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--pool", default="mean", choices=["mean", "cls0"])
a = ap.parse_args()

meta_p = os.path.join(a.cache, "meta.json")
meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
print(f"[meta] {meta}")
ids = np.load(os.path.join(a.cache, "ids.npy"))
N = len(ids)
size = os.path.getsize(os.path.join(a.cache, "feats.f16"))
per = size / 2 / N                                   # 每张图的 fp16 元素数
print(f"[in] {N} 张, feats.f16 {size / 1e9:.2f} GB, 每张 {per:.0f} 个元素")

# 从 meta 或候选表推 (T, D)
T = D = None
for k in ("tokens", "token", "n_tokens", "seq_len", "T"):
    if k in meta:
        T = int(meta[k])
for k in ("dim", "feature_dim", "d_model", "D"):
    if k in meta:
        D = int(meta[k])
if T is None or D is None:
    for ct, cd in ((256, 384), (256, 768), (1024, 384), (1369, 384), (324, 384), (1024, 768)):
        if abs(ct * cd - per) < 1:
            T, D = ct, cd
            break
if T is None:
    raise SystemExit(f"[err] meta 缺 tokens/dim 且无法从每张 {per} 个元素推出 (T,D); "
                     f"meta={meta}")
print(f"[shape] 推断 (T,D) = ({T},{D}), pool={a.pool}")

# 与 csv 的 img_id / callig / script 对齐
row_by_id = {}
for r in csv.DictReader(open(a.csv, encoding="utf-8")):
    row_by_id[int(r["img_id"])] = (int(r["calligrapher_id"]), int(r["script_id"]))
feat = np.zeros((N, D), np.float32)
calligs = np.full(N, -1, np.int64)
scripts = np.full(N, -1, np.int64)
n_match = 0
mm = np.memmap(os.path.join(a.cache, "feats.f16"), dtype=np.float16, mode="r")
mm = mm.reshape(N, T, D)
for s in range(0, N, a.chunk):
    e = min(s + a.chunk, N)
    blk = np.asarray(mm[s:e], np.float32)
    feat[s:e] = blk[:, 0] if a.pool == "cls0" else blk.mean(1)
    for i in range(s, e):
        cs = row_by_id.get(int(ids[i]))
        if cs:
            calligs[i], scripts[i] = cs
            n_match += 1
    if s % (a.chunk * 10) == 0:
        print(f"  pool {e}/{N}", flush=True)

keep = calligs >= 0
print(f"[align] csv 命中 {n_match}/{N}; 保留 {int(keep.sum())}")
uniq = len(np.unique(np.round(feat[keep][:2000], 4), axis=0))
print(f"[guard] 采样唯一特征数 = {uniq} (build 的守卫要求 >100)")
# ★ [2026-10-04] 补 char/glyph 列 —— 字×书体表预训练需要按 glyph 分组的标签。
#   用 img_id 与 csv 做 join (与上面 calligs/scripts 的对齐方式同构)。
#   顺序必须与 feat[keep] 一致: 缓存行 i 对应 ids[keep][i]。
#   ⚠ 本脚本是**模块级**的(无缩进), 插入块也必须顶格 —— 上次按 4 空格插导致
#      IndentationError, 已由语法自检拦下并回滚, 这次改对。
_glyph_by_img = {}
for _r in csv.DictReader(open(a.csv, encoding="utf-8")):
    try:
        _iid = int(_r["img_id"])
        _gid = int(_r["glyph_id"])
    except (KeyError, TypeError, ValueError):
        continue
    _glyph_by_img[_iid] = _gid
glyph_ids = np.array([_glyph_by_img.get(int(_i), -1) for _i in ids[keep]],
                     dtype=np.int64)
print(f"[align] glyph 命中 {int((glyph_ids >= 0).sum())}/{len(glyph_ids)}")
np.savez_compressed(a.out, feat=feat[keep], calligs=calligs[keep], scripts=scripts[keep], glyph_ids=glyph_ids)
print(f"[out] {a.out}  feat={feat[keep].shape}")
