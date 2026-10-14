#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preencode_disc_data.py —— 判别器数据的**一次性离线预编码** ✓（训练中不挂 decoder ✓）。

产出 scratch/disc_lat_full.npz:
    z        [N,4,32,32] fp16  真迹 latent（VAE posterior.mode() × sf ✓ 确定性 ✓）
    writer   [N] str   书家（y_callig_raw 口径 ✓ 10 类 ✓）
    script   [N] str   书体
    char     [N] str   字
    work     [N] str   作品 id（作品级留出 ✓）
    std_idx  [N] int   该样本对应 std 条目的行号（-1=缺 ✓）
    z_std    [M,4,32,32] fp16  标准字形 latent（同 VAE ✓ mode ✓）
    g_std    [M,64,64] uint8  64×64 std 条件图（**固定 σ=0.65** ✓ 方案要求 ✓）
    std_script/std_char [M] str
    writers/scripts     元数据 ✓
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.style.bank import resolve, work_id                          # noqa: E402

SF = 0.18215
VAE_DIR = "data/pretrained/sd-vae-ft-ema"  # ★ 以 v84A run manifest 为准 ✓(flux_vae 是 16 通道 ✗ 已排除 ✓)


def _posix(p):
    return str(p).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="exp-std/csv/train.csv")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--out", default="scratch/disc_lat_full.npz")
    ap.add_argument("--sigma", type=float, default=0.65)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    roots = args.root or [".", "data", "exp-std"]

    from PIL import Image, ImageFilter
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR).to(dev).eval()

    def encode(paths, tag):
        out = np.zeros((len(paths), 4, 32, 32), np.float16)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(paths), args.batch):
                x = torch.stack([
                    torch.from_numpy(np.asarray(Image.open(_posix(p)).convert("RGB")))
                    .permute(2, 0, 1).float() / 127.5 - 1.0 for p in paths[i:i + args.batch]
                ]).to(dev)
                z = vae.encode(x).latent_dist.mode() * SF          # ★ mode 确定性 ✓
                out[i:i + len(x)] = z.half().cpu().numpy()
                if (i // args.batch) % 100 == 0:
                    print("  [%s] %d/%d (%.0fs)" % (tag, i + len(x), len(paths),
                                                    time.time() - t0), flush=True)
        print("[%s] 完成 %s (%.0fs)" % (tag, out.shape, time.time() - t0), flush=True)
        return out

    # ── 真迹 ──
    rows = []
    for r in csv.DictReader(open(args.csv, encoding="utf-8-sig")):
        if (r.get("aug") or "").strip():
            continue
        p = resolve(_posix(r.get("image_path", "")), roots)
        if not p:
            continue
        rows.append({"path": p,
                     "writer": (r.get("calligrapher") or "").strip(),
                     "script": (r.get("script") or "").strip(),
                     "char": (r.get("character") or "").strip(),
                     "work": work_id(r.get("src_image_path", ""),
                                      r.get("image_path", "")),
                     "std": resolve(_posix(r.get("std_path", "")), roots) or ""})
    print("[data] 真迹 %d" % len(rows), flush=True)
    z = encode([r["path"] for r in rows], "real")

    # ── std（按 (script,char) 去重 ✓）──
    seen, std_rows = set(), []
    row_std_idx = np.full(len(rows), -1, np.int64)
    for i, r in enumerate(rows):
        key = (r["script"], r["char"])
        if not r["std"] or not os.path.isfile(r["std"]):
            continue
        if key not in seen:
            seen.add(key)
            std_rows.append({"path": r["std"], "script": r["script"], "char": r["char"]})
    # 精确映射: 每个真迹行 → 它的 std 条目行号 ✓
    key2std = {r["script"] + "·" + r["char"]: j for j, r in enumerate(std_rows)}
    for i, r in enumerate(rows):
        row_std_idx[i] = key2std.get(r["script"] + "·" + r["char"], -1)
    print("[data] std 去重 %d（覆盖 %d/%d 真迹 ✓）"
          % (len(std_rows), int((row_std_idx >= 0).sum()), len(rows)), flush=True)

    z_std = encode([r["path"] for r in std_rows], "std")

    # g_std: 64×64 + **固定 σ blur** ✓（方案: 保证同字同书体 g 逐位相同 ✓）
    g_std = np.zeros((len(std_rows), 64, 64), np.uint8)
    for j, r in enumerate(std_rows):
        im = Image.open(_posix(r["path"])).convert("L").resize((64, 64), Image.BICUBIC)
        im = im.filter(ImageFilter.GaussianBlur(radius=args.sigma))
        g_std[j] = np.asarray(im)
    print("[data] g_std %s（σ=%.2f ✓）" % (g_std.shape, args.sigma), flush=True)

    # ★ OOD 探针 latent（brush 负样本 ✓ —— 修 T3 伪影欺骗 ✓）
    probe_imgs = {"white": Image.new("RGB", (256, 256), (255, 255, 255)),
                  "noise": Image.fromarray(np.random.RandomState(0).randint(
                      0, 256, (256, 256, 3)).astype(np.uint8))}
    z_probe = np.zeros((len(probe_imgs), 4, 32, 32), np.float16)
    for pi, (nm, im) in enumerate(probe_imgs.items()):
        x = (torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()
             .unsqueeze(0).to(dev) / 127.5 - 1.0)
        with torch.no_grad():
            z_probe[pi] = (vae.encode(x).latent_dist.mode() * SF).half().cpu().numpy()
    print("[probe] OOD latents %s ✓" % list(probe_imgs), flush=True)

    writers = sorted({r["writer"] for r in rows})
    scripts = sorted({r["script"] for r in rows})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        z=z, writer=np.array([r["writer"] for r in rows]),
        script=np.array([r["script"] for r in rows]),
        char=np.array([r["char"] for r in rows]),
        work=np.array([r["work"] for r in rows]),
        std_idx=row_std_idx,
        z_std=z_std, g_std=g_std,
        std_script=np.array([r["script"] for r in std_rows]),
        std_char=np.array([r["char"] for r in std_rows]),
        probe_names=np.array(list(probe_imgs)), z_probe=z_probe,
        writers=np.array(writers), scripts=np.array(scripts))
    print("[done] %s（z %s | z_std %s ✓）" % (args.out, z.shape, z_std.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
