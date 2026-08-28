#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 s15_ws_flow 200k ckpt 在 CPU 上生成 seen_samples。
直接复用 in_process_eval.py 的 run_show5 逻辑，但改为 CPU 推理。
关键：用 create_diffusion_or_flow(diffusion_type='flow') 而非 eval_auto 的 DDPM-only。
"""
import os, sys, json, torch
os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT/src")

from argparse import Namespace
CKPT_DIR = "5script/results/s15_ws_flow/20260826-133102-s15-ws-flow"
CFG = f"{CKPT_DIR}/resolved_config.json"
CKPT = f"{CKPT_DIR}/checkpoints/0200000.pt"
SEEN_CSV = "5script/show5_top30.csv"

cfg = json.load(open(CFG))
a = Namespace(**cfg)

# ── Build model ──
from auto_eval_cpu import build_model, load_ckpt_weights
print("Building model (CPU)...")
model = build_model(a, "cpu")

print(f"Loading ckpt: {CKPT}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
load_ckpt_weights(model, ckpt, "0200000.pt")
del ckpt
model.eval()

# ── Build cache (same as in_process_eval prepare_small_cache) ──
from in_process_eval import prepare_small_cache, _save_batch_pngs, load_eval_vae
print("Building seen5 cache...")
cache = prepare_small_cache(
    SEEN_CSV,
    getattr(a, 'img_root', '') or getattr(a, 'data_dir', '') or '',
    a.image_size,
    int(getattr(a, 'vae_downscale', 4)),
    int(getattr(a, 'latent_channels', 4)),
)
print(f"  seen5 cache: {cache['n']} samples")

# ── VAE ──
print("Loading VAE (CPU)...")
vae = load_eval_vae(a, "cpu")

# ── Flow/DDPM sampling ──
from diffusion import create_diffusion_or_flow
diffusion_type = getattr(a, 'diffusion_type', 'ddpm')
print(f"Creating diffusion (type={diffusion_type}, steps={a.eval_steps})...")
diffusion = create_diffusion_or_flow(str(a.eval_steps), diffusion_type=diffusion_type)

n = cache["n"]
lc = cache["latent_channels"]
ls = cache["latent_spatial"]
sf = cache["scaling_factor"]
conds = cache["conds"]
noise_all = cache["noise"]
gts_all = cache["gts"]
cfg_scale = float(getattr(a, 'eval_cfg', 4.0))

step_tag = "step0200000"
out_dir = os.path.join(CKPT_DIR, "checkpoints", "seen_samples", step_tag)
os.makedirs(out_dir, exist_ok=True)

print(f"Running CPU inference ({diffusion_type}, {a.eval_steps} steps, CFG {cfg_scale})...")
print(f"Output: {out_dir}")
torch.manual_seed(int(getattr(a, 'eval_seed', 0)))

import time
t0 = time.time()
with torch.no_grad():
    z = noise_all[:n]
    yc = torch.tensor([c[0] for c in conds[:n]], dtype=torch.long)
    yh = torch.tensor([c[1] for c in conds[:n]], dtype=torch.long)
    mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
    samples = diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, z,
        clip_denoised=False, model_kwargs=mk, device="cpu",
    )
    decoded = vae.decode(samples / sf).sample
    # Save
    _save_batch_pngs(decoded.float().cpu(), gts_all[:n].clone(),
                     conds, out_dir, 200000, 0)
    # samples.json
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"step": 200000, "n": n, "cfg": cfg_scale,
                   "ddim_steps": a.eval_steps, "diffusion_type": diffusion_type}, f)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s!")
print(f"Files: {os.listdir(out_dir)}")
