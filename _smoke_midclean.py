# -*- coding: utf-8 -*-
"""Mid-clean dataset smoke on GPU: build MCCDLatentDataset + one flow train step
with the refactored stack (src packages). Confirms the pipeline is launchable.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["XFORMERS_DISABLED"] = "1"
import torch
sys.stdout.reconfigure(encoding="utf-8")

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow
from src.utils import MCCDLatentDataset
from torch.utils.data import DataLoader

DEV = "cuda"
torch.manual_seed(0)

print("building MCCDLatentDataset (mid-clean) ...")
t0 = time.time()
ds = MCCDLatentDataset(
    csv_file="5script/train_mid_clean.csv",
    latent_shards_dir="final_latents_mid_clean",
    img_root="final_imgs_256",
    skel_root="final_skeleton_d3",
    image_size=256, load_skel=False, load_image=False,
    use_glyph_cond=False, is_train=True, structure_size=256,
)
print(f"dataset len={len(ds)} ({time.time()-t0:.1f}s)")
s = ds[0]
print("keys:", list(s.keys()))
print("latent:", tuple(s["latent"].shape), s["latent"].dtype)
print("y_callig:", s["y_callig"].item(), "y_char:", s["y_char"].item())

# one training step with flow
m = DiT_2Cond_models["DiT-2Cond-S/2"](
    num_calligraphers=1011, num_characters=35130,
    condition_fusion="factorized_add", callig_embed_dim=128,
    char_embed_dim=384, char_proj_mode="ln_only", freeze_char_table=True,
    use_checkpoint=False, learn_sigma=True).to(DEV)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
diff = create_diffusion_or_flow("", diffusion_type="flow")

loader = DataLoader(ds, batch_size=48, shuffle=True, num_workers=4)
batch = next(iter(loader))
x = batch["latent"].to(DEV)
yc = batch["y_callig"].to(DEV)
yh = batch["y_char"].to(DEV)
t = diff.sample_t(x.shape[0], DEV)
print(f"batch latent={tuple(x.shape)} yc∈[{yc.min().item()},{yc.max().item()}] "
      f"yh∈[{yh.min().item()},{yh.max().item()}] t={t[:3].tolist()}")
opt.zero_grad()
with torch.autocast("cuda", dtype=torch.bfloat16):
    ld = diff.training_losses(m, x, t, dict(y_callig=yc, y_char=yh))
    loss = ld["loss"].mean()
print(f"flow loss={loss.item():.4f}")
loss.backward()
opt.step()
print(f"step OK, mem={torch.cuda.memory_allocated()/1e9:.2f}G")
print("MID-CLEAN GPU SMOKE PASSED")