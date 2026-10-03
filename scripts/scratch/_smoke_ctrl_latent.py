# -*- coding: utf-8 -*-
"""Smoke: skel-latent dataset + ControlNet latent-cond forward (no VAE)."""
import os, sys, torch
sys.path.insert(0, os.getcwd())

from src.utils import MCCDLatentDataset
from src.model.controlnet import ControlNetDiT, load_main_model
from src.loss import create_diffusion_or_flow

# 1) dataset: skel_latent path
ds = MCCDLatentDataset(
    csv_file="assets/train_mid_common.csv",
    latent_shards_dir="data/latents/final_latents_mid_clean",
    img_root=None, skel_root="data/skel/final_skel3",
    skel_latent_shards_dir="data/skel/final_skel_latents_mid_common",
    image_size=256, load_canny=False, load_skel=False,
    is_train=True, preload=False, load_image=False,
    structure_size=256)
print("[smoke] dataset n =", len(ds))
it = ds[0]
print("[smoke] keys:", sorted(it.keys()))
print("[smoke] latent:", tuple(it['latent'].shape),
      "skel_latent:", tuple(it['skel_latent'].shape),
      "callig:", int(it['y_callig']), "char:", int(it['y_char']))
assert it['skel_latent'].shape == torch.Size([4, 32, 32]), "skel latent shape wrong"
assert it['skel_latent'].dtype == torch.float32

# 2) eval cache: skel_latent shards path
from src.eval.inference import make_eval_cache
cache = make_eval_cache(
    "assets/eval_strict_top6.csv", "data/imgs/final_imgs_256", "data/skel/final_skel3",
    256, 4, 8, 4, 0.18215,
    skel_latent_shards_dir="data/skel/final_skel_latents_eval")
sl = cache["skels_latent"]
print("[smoke] eval cache skels_latent:", tuple(sl.shape), "skels(png):", tuple(cache["skels"].shape))
assert sl.shape == torch.Size([4, 4, 32, 32]) and sl.abs().sum() > 0, "eval skel latent empty"

# 3) ControlNet forward with latent cond
torch.manual_seed(0)
main = load_main_model(
    model_name="DiT-2Cond-S/2",
    ckpt_path="assets/results/s19_midclean_s_flow/20260828-143711-s19-midclean-s-flow/checkpoints/0050000.pt",
    device="cpu",
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384,
    char_proj_mode="ln_only", freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25)
ctrl = ControlNetDiT(main, cond_in_channels=4, train_ctrl_only=True).eval()

B = 2
x = torch.randn(B, 4, 32, 32)
t = torch.tensor([0.5, 0.3])  # flow t in [0,1]
yc = torch.tensor([377, 672])
yh = torch.tensor([18, 157])
skel = torch.randn(B, 4, 32, 32) * 0.1  # skel latent (arbitrary scale)
diff = create_diffusion_or_flow("", diffusion_type="flow")
print("[smoke] diffusion type:", type(diff).__name__, "num_timesteps:", diff.num_timesteps)

with torch.no_grad():
    out = ctrl(x, t * 1000.0, yc, yh, cond=skel)
print("[smoke] forward(x,t,yc,yh,cond=skel) ->", tuple(out.shape), out.dtype)
# learn_sigma=True -> 2C 输出 (main.in_channels=4), 训练/采样取前 4 通道
assert out.shape == (B, 2 * 4, 32, 32), "expected learn_sigma 2C output"

# 4) loss path (mimic train_controlnet): cond drop + training_losses
ld = diff.training_losses(ctrl, x, t, dict(y_callig=yc, y_char=yh, cond=skel))
print("[smoke] training_losses loss:", ld["loss"].shape, float(ld["loss"].mean().item()))

# 5) CFG sampling path (1-step Euler via ddim_sample_loop)
with torch.no_grad():
    smp = diff.ddim_sample_loop(
        ctrl.forward_with_cfg, (B, 4, 32, 32),
        model_kwargs=dict(y_callig=yc, y_char=yh, cfg_scale=1.7, cond=skel),
        device="cpu")
print("[smoke] ddim_sample_loop(forward_with_cfg, cond=skel) ->", tuple(smp.shape))

print("\n=== SMOKE PASSED ===")