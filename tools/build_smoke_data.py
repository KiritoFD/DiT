"""Build a tiny synthetic dataset for local flow smoke test.

Creates:
  _smoke_data/train.csv      (32 rows: image_path, calligrapher, char)
  _smoke_data/eval.csv       (4 rows)
  _smoke_data/final_imgs_256/{0..35}.png  (random noise-ish 256x256)
  _smoke_data/final_latents/shard_00000.npz  (VAE-encoded latents, 4x32x32)

Uses the local sd-vae-ft-ema to encode so train.py's latent-cached path works.
"""
import os, sys, json, csv
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_smoke_data")
os.makedirs(os.path.join(ROOT, "final_imgs_256"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "final_latents"), exist_ok=True)

torch.manual_seed(0)
N_TRAIN, N_EVAL = 32, 4
img_size = 256

# 1. generate synthetic images (structured noise: faint strokes on paper bg)
def make_img(seed):
    rng = np.random.default_rng(seed)
    img = np.full((img_size, img_size, 3), 245, dtype=np.uint8)  # paper
    # a few dark "stroke" blobs
    for _ in range(6):
        cx, cy = rng.integers(40, img_size - 40, 2)
        r = rng.integers(8, 30)
        yy, xx = np.ogrid[:img_size, :img_size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        img[mask] = rng.integers(20, 90, size=3)  # ink color
    return img

rows = []
for i in range(N_TRAIN + N_EVAL):
    arr = make_img(1000 + i)
    Image.fromarray(arr).save(os.path.join(ROOT, "final_imgs_256", f"{i}.png"))
    rows.append({
        "image_path": f"final_imgs_256/{i}.png",
        "calligrapher": f"c{i % 8}",
        "script": str(i % 3),
        "character": str(i % 16),
        "calligrapher_id": str(i % 8),
        "script_id": str(i % 3),
        "character_id": str(i % 16),
    })

CSV_COLS = ["image_path", "calligrapher", "script", "character",
            "calligrapher_id", "script_id", "character_id"]
with open(os.path.join(ROOT, "train.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLS)
    w.writeheader()
    for r in rows[:N_TRAIN]:
        w.writerow(r)

with open(os.path.join(ROOT, "eval.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLS)
    w.writeheader()
    for r in rows[N_TRAIN:]:
        w.writerow(r)

# 2. VAE-encode to latents (4x32x32)
print("loading VAE...")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "pretrained_models", "sd-vae-ft-ema"))
vae.eval()
sf = 0.18215
latents = []
img_ids = []
with torch.no_grad():
    for i in range(N_TRAIN + N_EVAL):
        img = Image.open(os.path.join(ROOT, "final_imgs_256", f"{i}.png")).convert("RGB")
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        x = x.unsqueeze(0)
        z = vae.encode(x).latent_dist.sample().mul_(sf)
        latents.append(z.squeeze(0).numpy())
        img_ids.append(i)
latents = np.stack(latents).astype(np.float32)
img_ids = np.array(img_ids, dtype=np.int64)
print("latents:", latents.shape)

np.savez_compressed(os.path.join(ROOT, "final_latents", "shard_00000.npz"),
                    latents=latents, img_ids=img_ids)
print("saved shard:", os.path.join(ROOT, "final_latents", "shard_00000.npz"))

# 3. dino index stub: glyph [[script,char]] for chars 0..15 -> index in 384-d file is unused for smoke
# (train.py only injects if char_dino_embeddings exists; we point at the real 384 file but our
#  char ids 0..15 exist in the 20468 glyph list? not guaranteed -> skip dino injection by
#  using a config without char_dino_* in the smoke run.)
print("DONE")
