# -*- coding: utf-8 -*-
"""bench_cpu2.py — 16 样本 CPU eval 微基准 (快速定参).

用法: taskset -c <cores> python tools/diag/bench_cpu2.py <threads> [batch]
测量: ctrl 臂 16 样本 Heun50+CFG / base 臂 16 / VAE decode 16 → 外推 100+100 全量.
"""
import os, sys, time
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 32
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 16
import torch
torch.set_num_threads(NT)
print(f"threads={NT} batch={BATCH}", flush=True)

from src.model.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import make_eval_cache, load_eval_vae
from src.eval.cpu_sampler import heun_sample_cpu

ARCH = dict(norm_type="rms", mlp_type="swiglu", qk_norm=True, rope=True,
            rope_theta=100.0, attn_impl="sdpa")
COMMON = dict(device=torch.device("cpu"), num_calligraphers=1013,
              num_characters=35130, condition_fusion="factorized_add",
              callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
              freeze_char_table=True, learn_sigma=False, **ARCH)
dev = torch.device("cpu")

main = load_main_model(
    ckpt_path="assets/results/v8_3stage/A_main_final.pt", **COMMON)
main.eval()
c = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True,
                  injection="modulate", null_cond="gaussian")
ckd = torch.load(
    "assets/results/v8_3stage/v8b/20260902-234912-v8b-s31-ctrl/checkpoints/0035000.pt",
    map_location="cpu", weights_only=False)
sd = {k: v for k, v in (ckd.get("ema") or ckd.get("ctrl")).items()
      if not k.startswith("main.") and not k.startswith("_orig_mod.main.")}
c.load_state_dict(sd, strict=False)
c.eval()

cache = make_eval_cache("assets/eval_fame_strict_clean_v8.csv", "final_imgs_fame_v8",
                        None, 256, 16, 8, 4, 0.18215,
                        skel_latent_shards_dir="final_skel_latents_fame_1px_v8")
noise, conds, skels = cache["noise"], cache["conds"], cache["skels_latent"].float()

N = 16
t0 = time.time()
lat_c = heun_sample_cpu(c, noise[:N], conds[:N], 0.7, BATCH, skel=skels[:N], seed=0)
t_ctrl = time.time() - t0
t0 = time.time()
lat_b = heun_sample_cpu(c, noise[:N], conds[:N], 0.7, BATCH, skel=None, seed=0)
t_base = time.time() - t0

vae = load_eval_vae(dev, "pretrained_models/sd-vae-ft-ema")
t0 = time.time()
_ = vae.decode(lat_b[:4] / 0.18215).sample  # warm
t1 = time.time()
_ = vae.decode(lat_b[:N] / 0.18215).sample
t_dec = time.time() - t1

full = t_ctrl * (100 / N) + t_base * (100 / N) + t_dec * (300 / N)
print(f"ctrl16={t_ctrl:.1f}s base16={t_base:.1f}s dec16={t_dec:.1f}s "
      f"-> 100+100+300dec 全量估计 {full/60:.1f} min (threads={NT}, batch={BATCH})",
      flush=True)
