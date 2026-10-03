# -*- coding: utf-8 -*-
"""eval_render_base_poster.py — 评测训练前 Render 基模 (v34 30k) 并生成真实真迹全景海报
"""
import os, sys, time
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

dev = th.device("cuda")

ckpt_path = "exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt"
eval_cache = "data/top10_style23/eval_real200_cache.pt"
out_poster = "exp/v34_stage2_mix25/render_base_eval_real200_poster.png"

print("=================================================================")
print("【评测训练前 Render 基模 (v34 30k) 在 200 样本纯真迹上的真实表现】")
print("=================================================================")

from src.eval import model_io, inference
from src.eval.metrics import ssim_torch
from src.eval.in_mem_eval import _get_vae

# 1. 载入模型
model, _ = model_io.load_model_from_ckpt(ckpt_path, device=dev, use_ema=True)
model.eval()
print(f"[model] Render 基模载入成功 -> {ckpt_path}")

# 2. 载入评测集
vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
diff_eval = inference.build_diffusion(25, "flow")

cache = th.load(eval_cache, map_location="cpu", weights_only=False)
eval_rows = cache["rows"]
eval_noise = cache["noise"].to(dev)
eval_conds = cache["conds"]
eval_std_lats = cache["std_lats"].to(dev)
eval_gt_skel_pngs = cache["gt_pngs"].to(dev)
eval_std_pngs = cache["std_pngs"].to(dev)
n_eval = len(eval_rows)

# 载入真迹原图
import torchvision.transforms as T
tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
eval_gt_imgs = []
for r in eval_rows:
    p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
    eval_gt_imgs.append(tf(Image.open(p).convert("RGB")))
eval_gt_imgs = th.stack(eval_gt_imgs).to(dev)
eval_gt_imgs_norm = (eval_gt_imgs + 1.0) / 2.0

# 3. 采样并出图
with th.no_grad():
    # 以 GT 骨架 (gt_lats) 作为条件
    gt_lats = cache["gt_lats"].to(dev)
    x_pred = inference.sample_latents(
        model, diff_eval, eval_noise, eval_conds,
        cfg_scale=1.0, batch=50, device=dev, skel=gt_lats
    )
    dec_list = []
    for s in range(0, n_eval, 28):
        _dec = (vae.decode(x_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
        dec_list.append(_dec)
    dec = th.cat(dec_list, dim=0)

    ssim_vals = ssim_torch(dec, eval_gt_imgs_norm).cpu().numpy()
    mean_ssim = float(np.mean(ssim_vals))
    med_ssim = float(np.median(ssim_vals))
    l1_val = float(th.nn.functional.l1_loss(dec, eval_gt_imgs_norm).item())

print(f"\n【训练前 Render 基模 (v34 30k) 评测结果】:")
print(f"  出墨 SSIM (均值)   : {mean_ssim:.4f} (中位: {med_ssim:.4f})")
print(f"  出墨 L1 误差       : {l1_val:.4f}\n")

# 4. 渲染 20 样本 3 行全景海报
p_cols = 20
p_canvas = Image.new("RGB", (256 * p_cols, 256 * 3))
sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
for col_idx, col in enumerate(sub_indices):
    p_skel = Image.fromarray((eval_gt_skel_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    p_ren = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    p_gt = Image.fromarray((eval_gt_imgs_norm[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    p_canvas.paste(p_skel, (col_idx * 256, 0))
    p_canvas.paste(p_ren, (col_idx * 256, 256))
    p_canvas.paste(p_gt, (col_idx * 256, 512))

p_canvas.save(out_poster)
print(f"[poster] 训练前 Render 基模全景对比海报已保存到: {out_poster}")
