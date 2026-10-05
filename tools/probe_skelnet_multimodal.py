"""判别实验: v38 skelnet 到底是"塌成条件均值"还是"采到了另一个合法模式"。

判据 (不引入 char y, 条件保持 w7 可观):
  · diversity  = 同一条件 K 次采样的两两 IoU    -> 接近 1 = 确定性(均值解), 低 = 真多峰
  · best-of-K  = K 次里与真迹最像的那个的 IoU  -> 显著高于单次 = 多峰 + 判据不对
  · cfg 扫描   = cfg>1 是否把 IoU/SSIM 拉回来 -> 是 = 推理端缺 guidance, 不是监督坏

用法: python tools/probe_skelnet_multimodal.py
"""
import os
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
dev = th.device("cuda")

CKPT = "exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt"
CACHE = "data/top10_style23/eval_real200_cache.pt"
N, K = 20, 4

from src.model.dit import DiT_2Cond_S_2
from src.eval import inference
from src.eval.in_mem_eval import _get_vae

model = DiT_2Cond_S_2(
    callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
    condition_fusion="factorized_cat", cond_fusion_norm="split",
    glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
    use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
    glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
    qk_norm=1, rope=1, rope_theta=100.0, use_checkpoint=False).to(dev).eval()

d = th.load(CKPT, map_location="cpu", weights_only=False)
sd = d.get("model", d)
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
model.load_state_dict(sd)
print(f"[model] loaded {CKPT}")

vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
diff_eval = inference.build_diffusion(25, "flow")
cache = th.load(CACHE, map_location="cpu", weights_only=False)
conds = list(cache["conds"])[:N]
std_lats = cache["std_lats"][:N].to(dev)
gt_pngs = cache["gt_pngs"][:N].to(dev)
std_pngs = cache["std_pngs"][:N].to(dev)


def masks(imgs):
    return (imgs.mean(1) < 0.6)


def iou(a, b):
    inter = (a & b).sum(-1).sum(-1).float()
    union = (a | b).sum(-1).sum(-1).float().clamp(min=1)
    return inter / union


gt_m = masks(gt_pngs)
std_m = masks(std_pngs)
base_std = iou(std_m, gt_m).mean().item()
print(f"[baseline] IoU(std, gt) = {base_std:.4f}   (生成必须超过它, 否则=不如不做)\n")
print(f"{'cfg':>5} | {'IoU单':>7} {'IoU最佳':>7} {'diversity':>9} | {'SSIM单':>7} {'SSIM最佳':>8}")

for cfg in (1.0, 1.5, 2.0, 3.0):
    th.manual_seed(1234)
    noise = th.randn(N * K, 4, 32, 32, device=dev)
    c2 = conds * K
    s2 = std_lats.repeat(K, 1, 1, 1)
    with th.no_grad():
        g = inference.sample_latents(model, diff_eval, noise, c2, cfg_scale=cfg,
                                     batch=20, device=dev, skel=s2)
        decs = []
        for s in range(0, g.shape[0], 20):
            decs.append(((vae.decode(g[s:s + 20].to(dev) / 0.18215).sample.clamp(-1, 1)) + 1) / 2)
        dec = th.cat(decs, 0)
    m = masks(dec).view(K, N, 256, 256)
    gt2 = gt_m.unsqueeze(0)                        # (1,N,256,256)
    ious = iou(m, gt2.expand(K, N, 256, 256))      # (K,N)
    ssims = th.stack([
        inference.ssim_torch(dec[k * N:(k + 1) * N], gt_pngs) for k in range(K)]).view(K, N) \
        if hasattr(inference, "ssim_torch") else None
    # diversity: 同一条件 K 次采样两两 IoU
    div = []
    for n in range(N):
        for a in range(K):
            for b in range(a + 1, K):
                div.append(iou(m[a, n], m[b, n]).item())
    iou_mean = float(ious.mean())
    iou_best = float(ious.max(0).values.mean())
    s_mean = float(ssims.mean()) if ssims is not None else float("nan")
    s_best = float(ssims.max(0).values.mean()) if ssims is not None else float("nan")
    print(f"{cfg:>5.1f} | {iou_mean:>7.4f} {iou_best:>7.4f} {np.mean(div):>9.4f} | "
          f"{s_mean:>7.4f} {s_best:>8.4f}")
print("\n判读: diversity≈1 -> 确定性均值解(要改监督/CFG); "
      "best>>单 -> 多峰判据问题; cfg>1 明显变好 -> 推理端缺 guidance")
