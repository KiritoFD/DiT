import os, sys, glob
import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

dev = th.device("cuda")

# Find the latest checkpoint from v37
ckpt_files = sorted(glob.glob("exp/v37_skelnet_sp/*/checkpoints/0002500.pt"))
if not ckpt_files:
    ckpt_files = sorted(glob.glob("exp/v37_skelnet_sp/*/checkpoints/*.pt"))
ckpt_path = ckpt_files[-1]
print(f"Loading checkpoint: {ckpt_path}")

from src.model.dit import DiT_2Cond_Sp_2
model = DiT_2Cond_Sp_2(
    callig_embed_dim=128,
    glyph_vec_cond=True,
    glyph_vec_dim=128,
    condition_fusion="factorized_cat",
    cond_fusion_norm="split",
    glyph_inject_layers=4,
    glyph_embedder_depth=2,
    num_calligraphers=23,
    use_glyph_cond=True,
    use_char_cond=False,
    learn_sigma=False,
    glyph_scale_init=0.6,
    norm_type="rms",
    mlp_type="swiglu",
    qk_norm=1,
    rope=1,
    rope_theta=100.0
).to(dev).eval()

d = th.load(ckpt_path, map_location="cpu", weights_only=False)
sd = d.get("ema", d.get("model", d))
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
model.load_state_dict(sd)

# Load eval cache
cache = th.load("data/top10_style23/eval_real200_cache.pt", weights_only=False)
noise = cache["noise"][:4].to(dev)
conds = cache["conds"][:4]
std_lats = cache["std_lats"][:4].to(dev)
gt_lats = cache["gt_lats"][:4].to(dev)
gt_pngs = cache["gt_pngs"][:4].to(dev)

from src.eval import inference
from src.eval.in_mem_eval import _get_vae
from src.eval.metrics import ssim_torch

vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
diff = inference.build_diffusion(25, "flow")

# Test 1: sample_latents with cfg_scale=0.0 (plain forward, no CFG wrapper)
print("\n--- Test 1: sample_latents cfg_scale=0.0 ---")
with th.no_grad():
    g1 = inference.sample_latents(model, diff, noise, conds, cfg_scale=0.0, batch=4, device=dev, skel=std_lats)
print(f"g1 latents: shape={g1.shape}, mean={g1.mean():.4f}, std={g1.std():.4f}, min={g1.min():.4f}, max={g1.max():.4f}")
print(f"gt_lats   : shape={gt_lats.shape}, mean={gt_lats.mean():.4f}, std={gt_lats.std():.4f}, min={gt_lats.min():.4f}, max={gt_lats.max():.4f}")
print(f"cos(g1, gt_lats): {th.cosine_similarity(g1.to(dev).flatten(1), gt_lats.flatten(1), dim=1)}")

dec1 = (vae.decode(g1.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
print(f"dec1: shape={dec1.shape}, min={dec1.min():.4f}, max={dec1.max():.4f}, mean={dec1.mean():.4f}")
print(f"gt_pngs: shape={gt_pngs.shape}, min={gt_pngs.min():.4f}, max={gt_pngs.max():.4f}, mean={gt_pngs.mean():.4f}")
ssim1 = ssim_torch(dec1, gt_pngs).cpu().numpy()
print(f"SSIM (cfg=0.0): {ssim1}, mean={np.mean(ssim1):.4f}")

# Test 2: sample_latents with cfg_scale=1.0
print("\n--- Test 2: sample_latents cfg_scale=1.0 ---")
with th.no_grad():
    g2 = inference.sample_latents(model, diff, noise, conds, cfg_scale=1.0, batch=4, device=dev, skel=std_lats)
dec2 = (vae.decode(g2.to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
ssim2 = ssim_torch(dec2, gt_pngs).cpu().numpy()
print(f"SSIM (cfg=1.0): {ssim2}, mean={np.mean(ssim2):.4f}")

# Test 3: Direct 1-Step Tweedie from noise
print("\n--- Test 3: Direct 1-Step Tweedie from t=1 ---")
with th.no_grad():
    t_1 = th.full((4,), 1000.0, device=dev)
    yc = th.tensor([c[0] for c in conds], device=dev, dtype=th.long)
    v_1 = model(noise, t_1, y_callig=yc, y_char=None, g=std_lats)
    g3 = noise - v_1
dec3 = (vae.decode(g3 / 0.18215).sample.clamp(-1, 1) + 1) / 2
ssim3 = ssim_torch(dec3, gt_pngs).cpu().numpy()
print(f"SSIM (1-Step): {ssim3}, mean={np.mean(ssim3):.4f}")
