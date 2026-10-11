# -*- coding: utf-8 -*-
"""skel_branch_probe.py — std-skel 分支到底能不能"细到 skel"？直接实验

两个测试（同一批 GT，N=16）：
  (A) **字形往返的骨架保真**：encode/decode 真迹 → 对 GT 掩码与重建掩码各取 1px 骨架 → IoU + 连通分量数
  (B) **1px 骨架自身的往返**：把 GT 的 1px 骨架画成图 → encode/decode → 阈值化后测
        · 骨架像素召回率 (rec ∩ gt)/|gt|
        · 连通性是否维持（分量数之比）—— 这直接回答"latent 能否承载骨架级信息"
"""
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAES = [
    ("SD-VAE", "pretrained_models/sd-vae-ft-ema", "mode"),
    ("Calli-DINO", "experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv", "sample"),
    ("Calli-topband", "experiments/vae_fix_topband/calli_vae_final", "sample"),
    ("Calli-physics", "experiments/calli_vae_physics/calli_vae_final", "sample"),
]
N = 16
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_of(rgb_uint8):
    return (rgb_uint8.mean(axis=2) < 128)


def comps(m):
    return int(ndimage.label(m, structure=np.ones((3, 3)))[1])


def main():
    rows = [r for r in csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8"))
            if os.path.exists(r["image_path"])][:N]
    gts = np.stack([np.asarray(Image.open(r["image_path"]).convert("RGB").resize((256, 256), Image.BICUBIC))
                    for r in rows])
    gt_masks = np.stack([mask_of(g) for g in gts])
    gt_skels = np.stack([skeletonize(m) for m in gt_masks])
    print(f"  N={N} 张; GT 平均骨架像素 = {gt_skels.sum(axis=(1,2)).mean():.0f}, "
          f"GT 平均连通分量 = {np.mean([comps(s) for s in gt_skels]):.1f}")
    print(f"\n  {'VAE':<14} | {'A: 字形往返 skelIoU':>18} | {'A: 分量比 rec/gt':>16} | "
          f"{'B: 1px骨架召回':>14} | {'B: 分量比 rec/gt':>16}")
    xt = torch.from_numpy(gts).float().permute(0, 3, 1, 2).to(DEV) / 127.5 - 1.0
    # (B) 把 1px 骨架渲染成图（黑线白底）—— 这就是"骨架条件"该给的东西
    skel_imgs = np.stack([np.where(s[..., None], 0, 255).repeat(3, axis=2).astype(np.uint8) for s in gt_skels])
    xs = torch.from_numpy(skel_imgs).float().permute(0, 3, 1, 2).to(DEV) / 127.5 - 1.0
    for name, path, mode in VAES:
        if not os.path.isdir(path):
            print(f"  {name:<14} | (缺 {path})")
            continue
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(path).to(DEV).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        outs = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for tag, x in (("glyph", xt), ("skel", xs)):
                d = vae.encode(x).latent_dist
                z = d.mode() if mode == "mode" else d.sample()
                outs[tag] = (((vae.decode(z).sample.float().clamp(-1, 1) + 1) / 2 * 255)
                             .permute(0, 2, 3, 1).cpu().numpy())
        # (A)
        ious, rat = [], []
        for k in range(N):
            rm = mask_of(outs["glyph"][k])
            rs = skeletonize(rm)
            inter = (rs & gt_skels[k]).sum()
            uni = (rs | gt_skels[k]).sum()
            ious.append(inter / max(uni, 1))
            rat.append(comps(rs) / max(comps(gt_skels[k]), 1))
        # (B)
        rec_b, rat_b = [], []
        for k in range(N):
            rm = mask_of(outs["skel"][k])
            rec_b.append((rm & gt_skels[k]).sum() / max(gt_skels[k].sum(), 1))
            rat_b.append(comps(rm) / max(comps(gt_skels[k]), 1))
        print(f"  {name:<14} | {np.mean(ious):>18.4f} | {np.mean(rat):>16.2f} | "
              f"{np.mean(rec_b):>14.4f} | {np.mean(rat_b):>16.2f}")
        del vae
        torch.cuda.empty_cache()
    print("""
  判读:
   * A: 字形往返后骨架 IoU —— 若明显 <1, 说明潜空间已丢失笔画拓扑（模型再怎么学也到不了骨架级）;
   * A: 分量比 >>1 = 断裂成多段（细笔断线）;
   * B: 把 1px 骨架本身送进去的召回率 —— 这是"latent 能否承载骨架条件"的直接上限;
        B 的分量比 >>1 = 细线被解码成断续虚影，骨架条件在潜空间里已被打散。
""")


if __name__ == "__main__":
    main()
