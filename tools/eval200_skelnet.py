"""eval200 统一三口径重评 (留出集, 与训练集 0 重叠)。

给 "std 骨架 -> 真迹骨架" 类模型 (skelnet) 用:
  · SSIM / 墨迹 SSIM / 骨架 IoU / LPIPS  四个口径
  · 每个口径都同时算 **"什么都不做(直接拿 std 骨架)"的基线** + ✓/✗
  · latent L2 (仅参考; 已证与结构不同源)
架构按 ckpt 的 state_dict 自动判定 glyph_concat_input (v44/v45 是 concat=True),
避免"加载静默跳过、指标照样算"的坑 (model_io 开头的警告)。

用法:
  python tools/eval200_skelnet.py --ckpt <ckpt.pt> [--n 200] [--steps 25] [--cfg 1.0]
"""
import argparse
import os
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
dev = th.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--cache", default="data/top10_style23/eval_real200_cache.pt")
ap.add_argument("--tag", default="")
a = ap.parse_args()

from src.eval import inference
from src.eval.metrics import ssim_torch
from src.eval.in_mem_eval import _get_vae, _lpips_per_sample
from src.model.dit import DiT_2Cond_S_2

ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
sd = ck.get("model") or ck.get("ema") or ck.get("gen_ema") or ck.get("gen") or ck
sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
# 自动判定 concat (x_embedder 首个卷积的输入通道 > 4 => 拼了骨架通道)
k0 = [k for k in sd if "x_embedder" in k and k.endswith("weight")]
concat = bool(k0) and sd[k0[0]].shape[1] > 4
# 自动判定 glyph_embedder 是否 depthwise-separable (v44/v45 开了, v38 没开)。
# 判据: 任一 glyph_embedder 卷积的 in_ch == 1 -> 深度可分离。
_sep = any(k.startswith("glyph_embedder") and k.endswith("weight")
           and v.dim() == 4 and v.shape[1] == 1 for k, v in sd.items())

model = DiT_2Cond_S_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_concat_input=concat,
    glyph_inject_layers=4, glyph_embedder_depth=2, glyph_embedder_sep=_sep,
    num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0, use_checkpoint=False).to(dev).eval()
ms, us = model.load_state_dict(sd, strict=False)
print(f"[model] {a.ckpt}\n   concat={concat} sep={_sep} "
      f"missing={len(ms)} unexpected={len(us)}")
if ms or us:
    print(f"   ⚠ missing={list(ms)[:4]} unexpected={list(us)[:4]}")

vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
diff = inference.build_diffusion(a.steps, "flow")
c = th.load(a.cache, map_location="cpu", weights_only=False)
N = min(a.n, len(c["img_ids"]))
conds = list(c["conds"])[:N]
noise = c["noise"][:N].to(dev)
std_lats = c["std_lats"][:N].to(dev)
gt_lats = c["gt_lats"][:N].to(dev)
std_pngs = c["std_pngs"][:N].to(dev)
gt_pngs = c["gt_pngs"][:N].to(dev)
print(f"[data] eval200 留出集 n={N} (与训练集 0 重叠)")

with th.no_grad():
    g = inference.sample_latents(model, diff, noise, conds, cfg_scale=a.cfg,
                                 batch=25, device=dev, skel=std_lats)
    decs = []
    for s in range(0, N, 25):
        decs.append(((vae.decode(g[s:s + 25].to(dev) / 0.18215).sample
                      .clamp(-1, 1)) + 1) / 2)
    dec = th.cat(decs, 0)


def masks(im):
    return im.mean(1) < 0.6


def iou(x, y):
    inter = (x & y).sum((-1, -2)).float()
    return (inter / (x | y).sum((-1, -2)).float().clamp(min=1)).mean().item()


def stat(name, pred_png, pred_lat=None):
    pm, gm = masks(pred_png), masks(gt_pngs)
    s = float(np.mean(ssim_torch(pred_png, gt_pngs).cpu().numpy()))
    ink = float(np.mean(ssim_torch(
        pm.float().unsqueeze(1).repeat(1, 3, 1, 1),
        gm.float().unsqueeze(1).repeat(1, 3, 1, 1)).cpu().numpy()))
    lp = _lpips_per_sample(pred_png.permute(0, 2, 3, 1).cpu().numpy(),
                           gt_pngs.permute(0, 2, 3, 1).cpu().numpy())
    lp = float(np.mean(lp)) if lp else float("nan")
    lat = (float((pred_lat.to(gt_lats.device).float() - gt_lats).pow(2).mean())
           if pred_lat is not None else float("nan"))
    row = dict(name=name, ssim=s, ink=ink, iou=iou(pm, gm), lpips=lp, l2=lat)
    return row


rows = [stat("std 基线(不做)", std_pngs, std_lats),
        stat("模型生成", dec, g)]
print(f"\n=== eval200 (n={N}) 三口径 + 基线 {a.tag} ===")
print(f"{'':<14}{'SSIM':>8}{'ink_ssim':>10}{'IoU':>8}{'LPIPS':>9}{'latentL2':>10}")
for r in rows:
    print(f"{r['name']:<14}{r['ssim']:>8.4f}{r['ink']:>10.4f}"
          f"{r['iou']:>8.4f}{r['lpips']:>9.4f}{r['l2']:>10.4f}")
b, m = rows[0], rows[1]
print(f"\n判读 (相对基线):")
print(f"  SSIM  {'✓' if m['ssim'] > b['ssim'] else '✗'}  "
      f"{m['ssim'] - b['ssim']:+.4f}")
print(f"  IoU   {'✓' if m['iou'] > b['iou'] else '✗'}  "
      f"{m['iou'] - b['iou']:+.4f}")
print(f"  LPIPS {'✓' if m['lpips'] < b['lpips'] else '✗'}  "
      f"{m['lpips'] - b['lpips']:+.4f}")
