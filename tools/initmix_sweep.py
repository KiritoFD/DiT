"""实验1: 混合初始点 (glyph_init_mix) 扫描 —— 不动训练, 只看"把起点锚到条件"能否立刻降 frag。

机制(仓库原有, 只是没接进 in_mem_eval): xT = alpha*noise + (1-alpha)*std_latent。
  alpha=1.0 -> 纯噪声 (现在的行为)
  alpha=0.0 -> 纯标准字 latent 起点
如果 frag 随 alpha 下降 -> "瘦/碎"里有一部分是**起点离条件太远**造成的, 便宜可修;
如果不动 -> 确认是训练期损失问题, 该上端点对比损失。

用法: python tools/initmix_sweep.py --alphas 1.0,0.8,0.6,0.4,0.2,0.0 --batch 4
"""
import argparse
import os
import sys
import time

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="exp-std/runs_purestd/20261003-163412-v46-std-adaln4-top10-p1.0"
                                   "/checkpoints/0025000.pt")
ap.add_argument("--alphas", default="1.0,0.8,0.6,0.4,0.2,0.0")
ap.add_argument("--csv", default="exp-std/csv/eval200_fixed.csv")
ap.add_argument("--n", type=int, default=187)
ap.add_argument("--shards", default="exp-std/data/shards_std_w7_fixed_eval200")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--vae-batch", type=int, default=4)
ap.add_argument("--cfg", type=float, default=0.7)
ap.add_argument("--steps", type=int, default=50)
a = ap.parse_args()

import src.eval.in_mem_eval as IME                                              # noqa: E402
from src.eval.inference import build_diffusion, sample_latents                  # noqa: E402
from src.eval.metrics import frag_ratio as _frag, hole_ratio as _hole           # noqa: E402
from src.eval.metrics import mse as _mse, ssim as _ssim                         # noqa: E402
from src.eval.metrics_ink import ink_iou as _ink_iou, ink_ssim as _ink_ssim     # noqa: E402
from src.eval.model_io import load_model_from_ckpt                              # noqa: E402

dev = th.device("cuda" if th.cuda.is_available() else "cpu")
alphas = [float(x) for x in a.alphas.split(",") if x.strip()]

model, args = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
args.eval_skel_latent_shards_dir = a.shards
args.skel_latent_shards_dir = "exp-std/data/shards_std_w7"
cache = IME._get_cache(a.csv, a.n, None, a.shards, args)
n = cache["n"]
g = cache["skels_latent"].cpu().float()          # 条件(std 骨架 latent) = 混合用的"起点锚"
noise = cache["noise"].cpu().float()
conds = cache["conds"]
vae = IME._get_vae(dev)
sf = float(getattr(args, "vae_scaling_factor", 0.18215))
gts = (cache["gts"].to(dev) + 1) / 2
DIFF = build_diffusion(a.steps, str(getattr(args, "diffusion_type", "flow")))
print(f"[in] n={n} cfg={a.cfg} steps={a.steps} alphas={alphas}")

rows = []
for al in alphas:
    z = al * noise + (1.0 - al) * g
    t0 = time.time()
    lat = sample_latents(model, DIFF, z, conds, a.cfg, a.batch, dev,
                         skel=cache["skels_latent"], seed=0)
    preds = th.empty_like(gts)
    for i in range(0, n, a.vae_batch):
        j = min(i + a.vae_batch, n)
        with th.autocast("cuda", dtype=th.bfloat16):
            dec = vae.decode(lat[i:j].to(dev) / sf).sample
        preds[i:j] = (dec.clamp(-1, 1) + 1) / 2
    p = preds.cpu().numpy().transpose(0, 2, 3, 1)
    gt = gts.cpu().numpy().transpose(0, 2, 3, 1)
    m = {
        "ssim": float(np.mean([_ssim(p[i], gt[i]) for i in range(n)])),
        "mse": float(np.mean([_mse(p[i], gt[i]) for i in range(n)])),
        "ink_ssim": float(np.mean([_ink_ssim(p[i], gt[i]) for i in range(n)])),
        "ink_iou": float(np.mean([_ink_iou(p[i], gt[i]) for i in range(n)])),
        "frag": float(np.mean([_frag(p[i], gt[i], thresh=0.5) for i in range(n)])),
        "hole": float(np.mean([_hole(p[i], thresh=0.5) for i in range(n)])),
    }
    rows.append((al, m))
    th.cuda.empty_cache()
    print(f"[alpha={al:4.1f}] ssim={m['ssim']:.4f} ink_ssim={m['ink_ssim']:.4f} "
          f"ink_iou={m['ink_iou']:.4f} frag={m['frag']:.3f} hole={m['hole']:.3f} "
          f"({time.time() - t0:.0f}s)")

print("\n============ 混合初始点扫描 (n=%d, ckpt step 25000, cfg=%.1f) ============" % (n, a.cfg))
print(f"{'alpha':>6} {'ssim':>8} {'ink_ssim':>9} {'ink_iou':>8} {'frag':>7} {'hole':>7}")
for al, m in rows:
    print(f"{al:>6.1f} {m['ssim']:>8.4f} {m['ink_ssim']:>9.4f} {m['ink_iou']:>8.4f} "
          f"{m['frag']:>7.3f} {m['hole']:>7.3f}")
print("\n参照: 什么都不做(std 直出) ssim=0.5990 ink_ssim=0.4337 ink_iou=0.1687 frag=1.019")
