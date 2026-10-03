"""算"什么都不做"基线: 把 sd 条件(std 骨架 latent, 模型真正看到的那个)解码后直接当输出,
用与 in_mem_eval **完全相同**的指标函数算分。

回答的问题: 我们这轮生成结果, 到底有没有超过"直接输出标准字"?
(in_mem_eval 里只有 pred vs GT 两组, 没有这个基线 -> 会出现"0.55 看着还行, 其实不如不做"的误判)
"""
import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.in_mem_eval import _get_cache, _get_vae, _lpips_per_sample  # noqa: E402
from src.eval.metrics import frag_ratio as _frag, hole_ratio as _hole  # noqa: E402
from src.eval.metrics import mse as _mse  # noqa: E402
from src.eval.metrics import ssim as _ssim  # noqa: E402
from src.eval.metrics_ink import ink_iou as _ink_iou  # noqa: E402
from src.eval.metrics_ink import ink_ssim as _ink_ssim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="eval200")
ap.add_argument("--csv", default="exp-std/csv/eval200_fixed.csv")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--shards", default="exp-std/data/shards_std_w7_fixed_eval200")
ap.add_argument("--ckpt", default="exp-std/runs_purestd/20261003-163412-v46-std-adaln4-top10-p1.0"
                                   "/checkpoints/0025000.pt", help="借它的 args 建 cache")
ap.add_argument("--vae-batch", type=int, default=16)
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
args = ck.get("args", {})
if isinstance(args, dict):
    args = Namespace(**args)

cache = _get_cache(a.csv, a.n, None, a.shards, args)
n = cache["n"]
print(f"[cache] set={a.set} n={n}  cond_shards={a.shards}")

vae = _get_vae(dev)
svae = float(getattr(args, "vae_scaling_factor", 0.18215))
gts = (cache["gts"].to(dev) + 1) / 2
preds = torch.empty_like(gts)
lat = cache["skels_latent"]                       # ← std 骨架条件 latent (模型输入)
for i in range(0, n, a.vae_batch):
    j = min(i + a.vae_batch, n)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dec = vae.decode(lat[i:j].to(dev) / svae).sample
    preds[i:j] = (dec.clamp(-1, 1) + 1) / 2

p = preds.cpu().numpy().transpose(0, 2, 3, 1)
g = gts.cpu().numpy().transpose(0, 2, 3, 1)

ss = np.array([_ssim(p[i], g[i]) for i in range(n)])
ms = float(np.mean([_mse(p[i], g[i]) for i in range(n)]))
lp = _lpips_per_sample(p, g, enabled=True)
iks = np.array([_ink_ssim(p[i], g[i]) for i in range(n)])
iki = np.array([_ink_iou(p[i], g[i]) for i in range(n)])
fr = np.array([_frag(p[i], g[i], thresh=0.5) for i in range(n)])
hp = np.array([_hole(p[i], thresh=0.5) for i in range(n)])
hg = np.array([_hole(g[i], thresh=0.5) for i in range(n)])

print(f"\n【{a.set} 什么都不做 (std 条件直出) 基线】n={n}")
print(f"  ssim      = {ss.mean():.4f}  (med={np.median(ss):.4f}, p10={np.percentile(ss, 10):.4f})")
print(f"  mse       = {ms:.5f}")
print(f"  lpips     = {float(np.mean(lp)):.4f}")
print(f"  ink_ssim  = {iks.mean():.4f}")
print(f"  ink_iou   = {iki.mean():.4f}")
print(f"  frag      = {fr.mean():.3f}")
print(f"  hole      = {hp.mean():.3f}/{hg.mean():.3f}")

out = f"exp-std/reeval/std_baseline_{a.set}.csv"
with open(out, "w", encoding="utf-8") as f:
    f.write("set,n,ssim,mse,lpips,ink_ssim,ink_iou,frag,hole_pred,hole_gt\n")
    f.write(f"{a.set},{n},{ss.mean():.4f},{ms:.5f},{float(np.mean(lp)):.4f},"
            f"{iks.mean():.4f},{iki.mean():.4f},{fr.mean():.3f},{hp.mean():.3f},{hg.mean():.3f}\n")
print(f"[out] {out}")
