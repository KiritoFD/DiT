#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skel_helpful.py — 最终判据: 在图像域比较 std / pred 骨架 与真实书法骨架的 SSIM。
S_gt  = data/50k/std/<oid>.png        (真实书法图 -> 已细化骨架, 这就是"真迹骨架")
S_std = VAE-decode(data/50k_v2_glyph15k/shards_std[oid])
S_pred= SkelNet(S_std, e_slot)
判据: mean_ssim(S_pred,S_gt) vs mean_ssim(S_std,S_gt)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.model.deform_skel import DeformSkel
from src.eval.in_mem_eval import _get_vae
from skimage.metrics import structural_similarity as ssim_fn

emb = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                 weights_only=False)["embedding"].float()
sd = torch.load("assets/deform_skel_top10_v1.pt", map_location="cpu", weights_only=False)
sd = sd.get("deform", sd)
sk = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=96,
                max_off=6.0, coarse=8, residual=0, res_cap=1.0,
                stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, dt_ch=1)
sk.load_state_dict(sd, strict=False)
sk = sk.to("cuda").eval()

vae = _get_vae("cuda")
SF = 0.18215
df = pd.read_csv("assets/eval_top10_strict_subset84.csv")
per = 5056

def load_std_latent(idx):
    s = idx // per; o = idx % per
    p = "data/50k_v2_glyph15k/shards_std/shard_%05d.npz" % s
    d = np.load(p)
    return torch.from_numpy(d["latents"][o:o+1]).float()

def decode(lat):
    with torch.no_grad():
        o = vae.decode(lat.to("cuda") / SF).sample
    return ((o.clamp(-1, 1) + 1) / 2)[0].mean(0).cpu().numpy()

def norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / max(1e-9, hi - lo)

def to128(a):
    if a.shape[0] != 128:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
            (128, 128), Image.Resampling.LANCZOS)).astype(np.float32) / 255.
    return a

res = []
for i, r in df.iterrows():
    oid = int(r["old_50k_id"]); pid = int(r["slot_id"])
    gt_p = "data/50k/std/%06d.png" % oid
    if not os.path.exists(gt_p):
        continue
    gt = np.asarray(Image.open(gt_p).convert("L").resize((128, 128),
                    Image.Resampling.LANCZOS)).astype(np.float32) / 255.
    gt = (gt > 0.5).astype(np.float32)          # 1=墨
    lat = load_std_latent(oid)
    std_img = decode(lat)                        # (H,W)
    std_img = (norm01(to128(std_img)) > 0.5).astype(np.float32)
    e = emb[pid:pid+1]
    with torch.no_grad():
        pred = sk(lat.to("cuda"), e.to("cuda"))
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
    pred_img = decode(pred.cpu())
    pred_img = (norm01(to128(pred_img)) > 0.5).astype(np.float32)

    s_std = ssim_fn(std_img, gt, data_range=1.0)
    s_pred = ssim_fn(pred_img, gt, data_range=1.0)
    res.append((i, r["character"], r["calligrapher"], pid, s_std, s_pred, s_pred - s_std))

d = pd.DataFrame(res, columns=["idx", "char", "cal", "pid", "ssim_std", "ssim_pred", "delta"])
print("=" * 78)
print("strict 84: 骨架与真实书法骨架的 SSIM (图像域, 二值)")
print("=" * 78)
print("  样本数:", len(d))
print("  S_std  vs S_gt : mean=%.4f median=%.4f" % (d.ssim_std.mean(), d.ssim_std.median()))
print("  S_pred vs S_gt : mean=%.4f median=%.4f" % (d.ssim_pred.mean(), d.ssim_pred.median()))
print("  Δ (pred-std)   : mean=%+.4f median=%+.4f" % (d.delta.mean(), d.delta.median()))
print("  正向提升比例    : %.1f%%" % (100 * (d.delta > 0).mean()))
print("  退化比例(<0)    : %.1f%%" % (100 * (d.delta < 0).mean()))
print()
print("  按书家:")
print(d.groupby("cal")["delta"].agg(["count", "mean"]).to_string())
print()
print("  最差 5 例:")
print(d.nsmallest(5, "delta")[["char", "cal", "ssim_std", "ssim_pred", "delta"]].to_string(index=False))
print("  最好 5 例:")
print(d.nlargest(5, "delta")[["char", "cal", "ssim_std", "ssim_pred", "delta"]].to_string(index=False))
