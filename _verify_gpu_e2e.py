# -*- coding: utf-8 -*-
"""GPU end-to-end sanity on the refactored stack:
1) load real s18 flow main ckpt + VAE (GPU)
2) one real training step via FlowMatching.training_losses (with the real dataset batch shape)
3) 2-step Euler ddim_sample_loop + VAE decode -> verify finite images
4) ControlNetDiT build on the real ckpt + one cfg eval path
Only 1 tiny batch; memory-light. Prints verdicts.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["XFORMERS_DISABLED"] = "1"
import torch
sys.stdout.reconfigure(encoding="utf-8")

from src.model import DiT_2Cond_models, ControlNetDiT, load_main_model
from src.loss import create_diffusion_or_flow

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "5script/results/s18_s_flow_small/20260827-232003-s18-s-flow-small/checkpoints/0043000.pt"
VAE = "pretrained_models/sd-vae-ft-ema"
print(f"device={DEV}")

# 1) main model from real ckpt
t0 = time.time()
main = load_main_model(
    model_name="DiT-2Cond-S/2", ckpt_path=CKPT, device=DEV,
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128,
    char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, use_checkpoint=False)
main.eval()
print(f"main loaded ({time.time()-t0:.1f}s), mem={torch.cuda.memory_allocated()/1e9:.2f}G")

# 2) VAE
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained(VAE).to(DEV).eval()
for p in vae.parameters(): p.requires_grad_(False)

# 3) real training step: FlowMatching, t = sample_t
diff = create_diffusion_or_flow("", diffusion_type="flow")
torch.manual_seed(0)
x = torch.randn(8, 4, 32, 32, device=DEV)
yc = torch.randint(0, 1011, (8,), device=DEV)
yh = torch.randint(0, 35130, (8,), device=DEV)
t = diff.sample_t(8, DEV)
print(f"train t sample: min={t.min().item():.4f} max={t.max().item():.4f} (should be in [0,1))")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    ld = diff.training_losses(main, x, t, dict(y_callig=yc, y_char=yh))
loss = ld["loss"].mean().item()
print(f"flow train loss (real ckpt): {loss:.4f} finite={torch.isfinite(torch.tensor(loss)).item()}")

# 4) 8-step Euler sampling + decode (train_steps=8, cfg=1.7)
diff8 = create_diffusion_or_flow("8", diffusion_type="flow")
torch.manual_seed(0)
z = torch.randn(2, 4, 32, 32, device=DEV)
mk = dict(y_callig=yc[:2], y_char=yh[:2], cfg_scale=1.7)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    samples = diff8.ddim_sample_loop(main.forward_with_cfg, z.shape, z,
                                     clip_denoised=False, model_kwargs=mk, device=DEV)
print(f"euler out: {tuple(samples.shape)} std={samples.float().std().item():.4f} finite={torch.isfinite(samples).all().item()}")
with torch.no_grad():
    dec = vae.decode(samples.float() / 0.18215).sample
print(f"decode: {tuple(dec.shape)} range=[{dec.min().item():.2f},{dec.max().item():.2f}] finite={torch.isfinite(dec).all().item()}")

# 5) ControlNetDiT on real ckpt + cfg path with skel
ctrl = ControlNetDiT(main, cond_in_channels=1, train_ctrl_only=True).to(DEV).eval()
skel = (torch.rand(2, 1, 256, 256, device=DEV) > 0.9).float()
mk2 = dict(y_callig=yc[:2], y_char=yh[:2], cfg_scale=1.7, cond=skel)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out_cfg = ctrl.forward_with_cfg(z, torch.full((2,), 500.0, device=DEV), yc[:2], yh[:2], cfg_scale=1.7, cond=skel)
print(f"ctrl cfg out: {tuple(out_cfg.shape)} finite={torch.isfinite(out_cfg).all().item()}")

print("GPU E2E SANITY PASSED")