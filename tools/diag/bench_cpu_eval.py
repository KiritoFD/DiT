# -*- coding: utf-8 -*-
"""bench_cpu_eval.py — CPU eval 吞吐基准 (定 batch/线程, 验证 2.5k 步间隙可行性).

用法: OMP_NUM_THREADS=56 python tools/diag/bench_cpu_eval.py [n_threads]
测量: ctrl 臂采样 (Heun50+CFG, 每 batch) / base 臂 / VAE decode → 外推 100+100 全量。
"""
import os, sys, time
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 56
import torch
torch.set_num_threads(NT)
try:
    torch.set_num_interop_threads(4)
except Exception:
    pass
print(f"threads={NT} torch={torch.__version__} phys_parallelism ok", flush=True)

from src.model.legacy.controlnet import load_main_model, ControlNetDiT
from src.eval.inference import make_eval_cache, sample_latents, load_eval_vae
from src.loss import create_diffusion_or_flow

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
ckd = __import__("torch").load(
    "assets/results/v8_3stage/v8b/20260902-234912-v8b-s31-ctrl/checkpoints/0035000.pt",
    map_location="cpu", weights_only=False)
sd = {k: v for k, v in (ckd.get("ema") or ckd.get("ctrl")).items()
      if not k.startswith("main.") and not k.startswith("_orig_mod.main.")}
c.load_state_dict(sd, strict=False)
c.eval()
vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
flow = create_diffusion_or_flow("50", diffusion_type="flow",
                                **{"t_sampler": "logit_normal", "sampler": "heun",
                                   "t_mean": 0.0, "t_std": 1.0, "shift": 1.0})
cache = make_eval_cache("assets/eval_fame_strict_clean_v8.csv", "data/imgs/final_imgs_fame_v8", None,
                        256, 64, 8, 4, 0.18215,
                        skel_latent_shards_dir="data/skel/final_skel_latents_fame_1px_v8")
noise, conds, skels = cache["noise"], cache["conds"], cache["skels_latent"].float()

for BATCH in (8, 16, 32):
    t0 = time.time()
    lat = sample_latents(c, flow, noise[:BATCH], conds[:BATCH], 0.7, BATCH, dev,
                         skel=skels[:BATCH], seed=0)
    t_ctrl = time.time() - t0
    t0 = time.time()
    lat2 = sample_latents(c, flow, noise[:BATCH], conds[:BATCH], 0.7, BATCH, dev,
                          skel=None, seed=0)
    t_base = time.time() - t0
    est = t_ctrl * (100 / BATCH) + t_base * (100 / BATCH)
    print(f"batch={BATCH:>3}: ctrl {t_ctrl:.1f}s  base {t_base:.1f}s  "
          f"-> 100+100 采样估计 {est:.0f}s", flush=True)

B = 16
lat = sample_latents(c, flow, noise[:B], conds[:B], 0.7, B, dev, skel=None, seed=0)
t0 = time.time()
_ = vae.decode((lat[:B] * 0.18215) / 0.18215).sample  # warm
t1 = time.time()
dec = vae.decode(lat[:B] / 0.18215).sample
t_dec = time.time() - t1
print(f"vae decode batch={B}: {t_dec:.1f}s -> 300 decodes 估计 {t_dec*300/B:.0f}s", flush=True)
print("done", flush=True)
