#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""batch_test_fp16.py — fp16 推理测试最大 batch。"""
import os, sys, time, json, glob
import numpy as np
import torch
sys.stdout.reconfigure(encoding="utf-8")

BASE = "/root/Workspace/xy/DiT"
os.chdir(BASE)

def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# 加载 ckpt args
ckpt_dir = "5script/results/s10_b4_grey_clear/20260824-151758-s10-b4-grey-clear/checkpoints"
first_ckpt = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))[0].replace(".done", "")
ckpt = torch.load(first_ckpt, map_location="cpu", weights_only=False)
args = ckpt["args"]
del ckpt

device = "cuda"

# Build model in fp16
from models import DiT_2Cond_models
vae_downscale = getattr(args, "vae_downscale", 4)
latent_size = args.image_size // vae_downscale
lc = int(getattr(args, "latent_channels", 4))
ls = latent_size
sf = float(getattr(args, "vae_scaling_factor", 0.18215))

model_cls = DiT_2Cond_models[args.model]
model = model_cls(
    input_size=latent_size, in_channels=lc,
    num_calligraphers=getattr(args, "num_calligraphers", 1011),
    num_characters=getattr(args, "num_characters", 7765),
    condition_fusion=getattr(args, "condition_fusion", "factorized_add"),
    callig_embed_dim=int(getattr(args, "callig_embed_dim", 128)),
    char_embed_dim=int(getattr(args, "char_embed_dim", 512)),
    learn_sigma=True,
    cond_drop_all_prob=float(getattr(args, "cond_drop_all_prob", 0.05)),
    cond_drop_one_prob=float(getattr(args, "cond_drop_one_prob", 0.25)),
).to(device).eval()

# Load EMA weights
ckpt = torch.load(first_ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["ema"], strict=False)
del ckpt
log(f"[model] loaded, fp32 params")

# VAE
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
log(f"[vae] loaded")

# DINO injection
emb = np.load(args.char_dino_embeddings)
idx_data = json.load(open(args.char_dino_index))
glyphs = idx_data.get("glyphs", idx_data)
NUM_CH = 7026
with torch.no_grad():
    for gi, (sid, cid) in enumerate(glyphs):
        gid = int(sid) * NUM_CH + int(cid)
        if 0 <= gid < model.y_char_embedder.embedding_table.weight.shape[0] and gi < emb.shape[0]:
            e = emb[gi]
            e = e / (np.linalg.norm(e) + 1e-8)
            model.y_char_embedder.embedding_table.weight.data[gid] = torch.from_numpy(e).float()
log(f"[dino] injected")

# Convert to fp16
model.half()
vae.half()
log(f"[model+vae] converted to fp16")

from diffusion import create_diffusion
ddim = create_diffusion("50")

# Batch test
for bs in [64, 128, 192, 256, 384, 455]:
    if bs > 455:
        bs = 455
    log(f"[test] batch={bs} fp16...")
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            z = torch.randn(bs, lc, ls, ls, device=device, dtype=torch.float16)
            mk = dict(
                y_callig=torch.tensor([0]*bs, device=device),
                y_char=torch.tensor([0]*bs, device=device),
                cfg_scale=4.0,
            )
            samples = ddim.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                            clip_denoised=False, model_kwargs=mk, device=device)
            dec = vae.decode(samples / sf).sample
            peak = torch.cuda.max_memory_allocated() / 1e9
        log(f"  batch={bs}: OK! peak={peak:.2f}G")
        del z, samples, dec
        torch.cuda.empty_cache()
        if bs >= 455:
            break
    except torch.cuda.OutOfMemoryError:
        log(f"  batch={bs}: OOM!")
        torch.cuda.empty_cache()
        break
    except Exception as e:
        log(f"  batch={bs}: error: {e}")
        torch.cuda.empty_cache()
        break

log("done")
