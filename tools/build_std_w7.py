#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_std_w7.py — 造 7px 的**标准骨架** latent shards (shards_std_w7)。

为什么需要:
  训练是 (std 条件 -> w7 目标)。现有 shards_std 是 **3px**, 目标是 **7px**,
  模型被要求"一边形变一边加粗 2.3x" -> 白耗容量, 实测输出宽度≈输入宽度
  (std≈3.4px, 模型输出 2.6~3.7px, 目标 6.9px)。
  把条件也做成 7px, 任务就退化成**纯形变(等宽)**, 与 G(w3->w3) 臂同构
  —— 那一臂的潜在对齐度 resAlign 0.49 明显高于 w7 臂的 0.40, 正是这个道理。

做法 (与 GT 骨架同配方, 只有宽度不同):
  std latent --VAE decode--> 二值化(dec<0) --skeletonize--> 1px 中心线
            --3x3 膨胀 iters 次(3 -> 7px)--> 像素图 --VAE encode--> 新 latent

产物: npz{latents(f16), img_ids}, 与现有 shards 完全同构, 可直接当 --cond-shards。
每个 shard 独立落盘, 已存在则跳过 (可断点续跑)。

用法:
  python tools/build_std_w7.py --probe            # 只量宽度, 不生成
  python tools/build_std_w7.py --iters 3          # 全量生成
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/top10_style23/shards_std")
    ap.add_argument("--out", default="data/top10_style23/shards_std_w7")
    ap.add_argument("--iters", type=int, default=3,
                    help="3x3 膨胀次数: 1=3px, 2=5px, 3=7px (与 gt_skel_png_w7 同档)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 shard (调试)")
    ap.add_argument("--probe", action="store_true", help="只量宽度对照, 不生成")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from diffusers.models import AutoencoderKL
    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize
    st8 = np.ones((3, 3), bool)

    srcs = sorted(glob.glob(os.path.join(a.src, "shard_*.npz")))
    if a.limit:
        srcs = srcs[:a.limit]
    print(f"[1] 源 {a.src}: {len(srcs)} 个 shard; 膨胀 {a.iters} 次 -> "
          f"{2 * a.iters + 1}px; 输出 {a.out}")

    print("[2] 载入 VAE ...")
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    def dec(lat):
        with th.no_grad(), th.autocast("cuda", dtype=th.float16):
            return vae.decode((lat / 0.18215).half()).sample.mean(1).float().cpu().numpy()

    def enc(img_u8):
        v = th.from_numpy((img_u8.astype(np.float32) / 255.0)) * 2 - 1
        v = v[:, None].repeat(1, 3, 1, 1)
        zs = []
        for s in range(0, v.shape[0], 8):
            with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                zs.append(vae.encode(v[s:s + 8].to(dev)).latent_dist.mode())
            th.cuda.empty_cache()
        return th.cat(zs) * 0.18215

    def thicken(ink):
        """(B,256,256) bool 墨 -> 骨架化 + 膨胀到目标宽度"""
        out = np.empty_like(ink)
        for i in range(ink.shape[0]):
            sk = skeletonize(ink[i]) if ink[i].any() else ink[i]
            out[i] = binary_dilation(sk, structure=st8, iterations=a.iters)
        return out

    # ── 宽度对照探针 (std / 膨胀后 / w7 目标) ──
    with np.load(srcs[0]) as z:
        lat0 = z["latents"][:16].astype(np.float32)
        ids0 = z["img_ids"][:16]
    t0 = th.from_numpy(lat0).to(dev)
    d0 = dec(t0)
    ink0 = d0 < 0
    th0 = thicken(ink0)
    print(f"\n[3] 宽度对照 (n={len(ids0)}, 墨占比; 3px 参考 = 膨胀1次):")
    print(f"    std 原样 (3px)           {ink0.mean():.5f}")
    print(f"    std 骨架化+膨胀{a.iters}次({2 * a.iters + 1}px)  {th0.mean():.5f}")
    gt = "data/top10_style23/shards_gtskel_w7"
    gs = sorted(glob.glob(os.path.join(gt, "shard_*.npz")))
    if gs:
        want = set(int(i) for i in ids0)
        got = {}
        for f in gs:
            with np.load(f) as z:
                for j, i in enumerate(z["img_ids"]):
                    if int(i) in want:
                        got[int(i)] = np.asarray(z["latents"][j], np.float32)
            if len(got) >= len(want):
                break
        if got:
            gk = list(got)
            dg = dec(th.from_numpy(np.stack([got[i] for i in gk])).to(dev))
            gink = dg < 0
            same = [k for k, i in enumerate(ids0) if int(i) in gk]
            ratio = (th0[same].mean() / max(gink.mean(), 1e-9)) if same else float("nan")
            print(f"    w7 目标 (训练目标本身)   {gink.mean():.5f}   "
                  f"(同 id {len(same)} 条: 膨胀后/目标 = {ratio:.2f}x)")
    if a.probe:
        print("\n--probe: 只量宽度, 不生成")
        return

    # ── 全量生成 ──
    os.makedirs(a.out, exist_ok=True)
    for si, sp in enumerate(srcs):
        op = os.path.join(a.out, os.path.basename(sp))
        if os.path.exists(op):
            print(f"[4] {si + 1}/{len(srcs)} 跳过 (已存在) {op}")
            continue
        with np.load(sp) as z:
            lat = z["latents"].astype(np.float32)
            ids = z["img_ids"].astype(np.int64)
        outs, inks = [], []
        for s in range(0, lat.shape[0], a.batch):
            b = lat[s:s + a.batch]
            d = dec(th.from_numpy(b).to(dev))
            ink = d < 0
            proc = thicken(ink)
            img = np.where(proc, 0, 255).astype(np.uint8)      # 墨=黑=0
            zz = enc(img).cpu().numpy().astype(np.float16)
            outs.append(zz)
            inks.append(float(proc.mean()))
            del d, ink, proc
            th.cuda.empty_cache()
        o = np.concatenate(outs)
        np.savez_compressed(op, latents=o, img_ids=ids)
        print(f"[4] {si + 1}/{len(srcs)} 写出 {op}  n={len(ids)}  "
              f"厚度墨占比 {float(np.mean(inks)):.5f}", flush=True)
    print("[5] DONE ->", a.out)


if __name__ == "__main__":
    main()
