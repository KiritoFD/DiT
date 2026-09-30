#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_skel64_dataset.py — 落盘 **64x64 骨架数据集** (用户 2026-09-30 裁定口径)。

口径 (严格按裁定): **原始图 -> 降到 64 -> 落盘 -> 再提 skel**
  · 降采样用 **min-pool** (4x4 块取最暗): 任何一块里有墨就算有墨 —— 细笔画不会
    像 block-mean 那样被平均成灰而消失。降采样后一条 1px 线在 256² 口径下约 4px。
  · 然后在 **64x64 上** skeletonize 提 1px 中心线 (不是在 256 上提完再降采样 ——
    顺序不同结果不同, 后者会把中心线在高分辨率下的分叉/细节带进来再压掉)。
  · 两张源图都做: 书法原迹 (imgs/<id>.png) 和 标准字图 (std/<id>.png)。

产物 (npz, 每 split 一个): ids / img_ink64 / std_ink64 / gt_skel64 / std_skel64
  全部 uint8 0/1, 形状 (N,64,64)。落盘后可反复复用, 训练不再需要 VAE 解码。

用法:
  python tools/build_skel64_dataset.py --res 64
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SPLITS = {
    "train": ("assets/train_top10_style23_minusval.csv", "data/top10_style23/std"),
    "val": ("assets/val_skelnet.csv", "data/top10_style23/std"),
    "seen20": ("assets/eval_top10_seen_20.csv", "data/top10_style23/std"),
    # strict84 的 std 源在 50k 集: 没有 png, 用 50k 的 shards_std 解码兜底
    "strict84": ("assets/eval_top10_strict_subset84.csv", ""),
}


def img_id_of(r):
    if r.get("img_id"):
        return int(r["img_id"])
    m = re.search(r"(\d+)\.png$", r["image_path"])
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--skel-width", type=int, default=1,
                    help="★ 骨架宽度(px@64)。1=纯 1px 中心线(墨~0.055, 稀疏难生成);\n"
                         "  3=膨胀1次(墨~0.15, 密3x, 好生成) —— **输入与目标同时用同一宽度**,\n"
                         "  保持'两边同宽更好学'的原则。末端骨架化后处理不变。")
    ap.add_argument("--out", default="data/top10_style23/skel64")
    ap.add_argument("--img-dir", default="data/top10_style23/imgs")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--std-shards-50k", default="data/50k_v2_glyph15k/shards_std")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    a = ap.parse_args()
    R = a.res
    K = 256 // R
    os.makedirs(a.out, exist_ok=True)

    from skimage.morphology import skeletonize
    from PIL import Image

    def minpool(g):
        return g.reshape(R, K, R, K).min(axis=(1, 3))

    _it = max(0, (a.skel_width - 1) // 2)
    from scipy.ndimage import binary_dilation as _bd

    def skel_feature(gray256):
        """原始图 (256 灰度) -> (ink64, skel64)。skel64 按 --skel-width 同宽膨胀。"""
        if gray256 is None:
            return np.zeros((R, R), np.uint8), np.zeros((R, R), np.uint8)
        ink = minpool(gray256) < 128
        if not ink.any():
            return ink.astype(np.uint8), np.zeros((R, R), np.uint8)
        sk = skeletonize(ink)
        if _it > 0:
            sk = _bd(sk, structure=np.ones((3, 3), bool), iterations=_it)
        return ink.astype(np.uint8), sk.astype(np.uint8)

    # strict84 的 std 用 50k shards decode 兜底 (只有 84 张, 可接受)
    _cache50 = {}
    _lc = __import__("threading").Lock()

    def std_from_shards50(iid):
      with _lc:                       # ⚠ 线程里懒加载 VAE 必须加锁, 否则会建多个
        if not _cache50:
            import torch as th
            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(a.vae).cuda().eval()
            mp = {}
            for sp in sorted(glob.glob(a.std_shards_50k + "/shard_*.npz")):
                with np.load(sp) as z:
                    for j, i in enumerate(z["img_ids"]):
                        mp[int(i)] = (sp, j)
            _cache50["idx"] = mp
            _cache50["vae"] = vae
        mp = _cache50["idx"]
        if iid not in mp:
            return None
        sp, j = mp[iid]
        with np.load(sp) as z:
            lat = np.asarray(z["latents"][j], np.float32)[None]
        import torch as th
        with th.no_grad(), th.autocast("cuda", dtype=th.float16):
            d = _cache50["vae"].decode((th.from_numpy(lat).cuda()
                                        / 0.18215).half()).sample.mean(1)
        b = (d[0] < 0).float().cpu().numpy()
        # 二值 latent 解码 -> 二值图; 转灰度口径 (墨=0)
        return np.where(b > 0.5, 0.0, 255.0).astype(np.uint8)

    for name, (csvp, stdd) in SPLITS.items():
        if not os.path.exists(csvp):
            print(f"[skip] {name}: {csvp} 不存在")
            continue
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
        ids, imgpaths, stdpaths = [], [], []
        for r in rows:
            i = img_id_of(r)
            if i is None:
                continue
            ip = r.get("image_path", "")
            if not os.path.isabs(ip) and not os.path.exists(ip):
                ip = os.path.join(a.img_dir, f"{i:06d}.png")
            sp_ = os.path.join(stdd, f"{i:06d}.png") if stdd else ""
            ids.append(i)
            imgpaths.append(ip)
            stdpaths.append(sp_)
        print(f"[{name}] {len(ids)} 条, 开始 (workers={a.workers}) ...", flush=True)
        t0 = time.time()

        def one(k):
            try:
                g_img = np.asarray(Image.open(imgpaths[k]).convert("L"))
                if g_img.shape != (256, 256):
                    g_img = None
            except Exception:                                  # noqa: BLE001
                g_img = None
            g_std = None
            if stdpaths[k] and os.path.exists(stdpaths[k]):
                try:
                    g_std = np.asarray(Image.open(stdpaths[k]).convert("L"))
                except Exception:                              # noqa: BLE001
                    g_std = None
            if g_std is None:
                g_std = std_from_shards50(ids[k])
            ii, gs = skel_feature(g_img)
            si, ss = skel_feature(g_std)
            return ii, si, gs, ss

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, range(len(ids))))
        img_ink = np.stack([r[0] for r in res])
        std_ink = np.stack([r[1] for r in res])
        gt_skel = np.stack([r[2] for r in res])
        std_skel = np.stack([r[3] for r in res])
        p = os.path.join(a.out, f"{name}.npz")
        # ★ 存 **0/255** (与 PNG / 全仓库图口径一致)。此前存 0/1 导致训练侧
        #   "把 0/1 当 0/255" 的静默 bug (背景被当成墨, 数据全废)。
        np.savez_compressed(p, ids=np.array(ids, np.int64), res=np.int64(R),
                            img_ink=img_ink * 255, std_ink=std_ink * 255,
                            gt_skel=gt_skel * 255, std_skel=std_skel * 255)
        print(f"[{name}] 落盘 {p}  {int(time.time()-t0)}s  (值域 0/255)\n"
              f"    64² 墨占比: 原迹 {img_ink.mean():.4f} / 标准字 {std_ink.mean():.4f}\n"
              f"    1px 骨架墨占比: GT {gt_skel.mean():.4f} / std {std_skel.mean():.4f}\n"
              f"    骨架为空: GT {(gt_skel.reshape(len(ids),-1).sum(1)==0).sum()} / "
              f"std {(std_skel.reshape(len(ids),-1).sum(1)==0).sum()}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
