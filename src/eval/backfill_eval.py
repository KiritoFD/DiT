# -*- coding: utf-8 -*-
"""
补齐前 10000 step 的 auto-eval 图片（历史只存了 eval_auto_*.json 数值，没存图）。
对 new_data 里 step 1000..10000 的每个 ckpt 加载 -> 对 eval 集跑 eval_in_memory -> 存 eval_<step>.png。
复用 train.py 的模型构建 + LoRA 注入 + delta 加载 + eval_cache 逻辑。
"""
import os, sys, glob, re, datetime
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import numpy as np

from src.model import DiT_3Cond_models
from src.loss import create_diffusion
from src.utils import find_model
from src.utils import MCCDDataset
from eval_auto import prepare_eval_cache, eval_in_memory
from src.model import inject_lora

BASE = "/root/Workspace/xy/DiT"
EXP = f"{BASE}/new_data/results_full_3cond/20260814-013816-DiT-3Cond-XL-2"
CKPT_DIR = f"{EXP}/checkpoints"
PRETRAINED = f"{BASE}/pretrained_models/DiT-XL-2-256x256.pt"
VAE_PATH = f"{BASE}/pretrained_models/sd-vae-ft-ema"
EVAL_CSV = f"{BASE}/final_eval.csv"

STEPS = list(range(1000, 11000, 1000))  # 1000..10000

MODEL = "DiT-3Cond-XL/2"
NUM_CALLIG = 1873
NUM_SCRIPT = 12
NUM_CHAR = 7765
LORA_R = 32
LORA_ALPHA = 32
EVAL_N = 1000
IMAGE_SIZE = 256


def main():
    assert torch.cuda.is_available(), "need GPU"
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda:0"
    latent_size = IMAGE_SIZE // 8

    # ---- model (build once, delta reloaded per ckpt) ----
    model = DiT_3Cond_models[MODEL](
        input_size=latent_size,
        num_calligraphers=NUM_CALLIG,
        num_scripts=NUM_SCRIPT,
        num_characters=NUM_CHAR,
        use_checkpoint=False,
    ).to(device)

    sd = find_model(PRETRAINED)
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("y_embedder", "y_callig", "y_script", "y_char", "cond_fusion"))}
    model.load_state_dict(sd, strict=False)

    model = inject_lora(model, r=LORA_R, lora_alpha=LORA_ALPHA)

    # ---- VAE + diffusion ----
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
    vae.eval()
    vae.requires_grad_(False)
    diffusion = create_diffusion(timestep_respacing="")
    # model 只需 forward；eval_in_memory 会用 diffusion 采样。需设 train()? eval 用 model(x,t,y)。

    # ---- eval cache ----
    print(f"[backfill] preparing eval cache n={EVAL_N} csv={EVAL_CSV} ...")
    _eval_ds = MCCDDataset(csv_file=EVAL_CSV, root_dir="",
                           image_size=IMAGE_SIZE, load_canny=False, load_skel=False)
    cache = prepare_eval_cache(vae, _eval_ds, device, n=EVAL_N)
    print("[backfill] eval cache ready.")

    model.eval()
    for step in STEPS:
        ckpt = f"{CKPT_DIR}/{step:07d}.pt"
        if not os.path.exists(ckpt):
            print(f"[backfill] MISSING {ckpt}, skip")
            continue
        out_dir = f"{EXP}/eval_{step:07d}"   # 目录，内含 1000 张 pred_*.png
        n_done = len(glob.glob(os.path.join(out_dir, "pred_*.png")))
        if n_done >= EVAL_N:
            print(f"[backfill] eval_{step:07d}/ already has {n_done} pngs, skip")
            continue
        # reload delta
        rf = torch.load(ckpt, map_location="cpu", weights_only=False)
        delta = rf.get("delta", rf.get("model", rf))
        missing, unexpected = model.load_state_dict(delta, strict=False)
        print(f"[backfill] step={step} loaded delta (missing={len(missing)})")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            mse, ss = eval_in_memory(model, vae, diffusion, device, cache,
                                     n=EVAL_N, vis_out=out_dir, vis_n=0)
        n_ok = len(glob.glob(os.path.join(out_dir, "pred_*.png")))
        print(f"[backfill] step={step}: MSE={mse:.5f} SSIM={ss:.4f} -> {n_ok} pngs in {out_dir}")
        # 释放中间
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
