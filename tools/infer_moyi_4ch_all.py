import os
import sys
import json
import csv
import time
import numpy as np
from PIL import Image

base = "/home/ds/Workspace/moyi/ref/moyi"
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, "moyun"))

import torch
from moyun_2 import DiT_models
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Load Vocab
vocab_path = "/home/ds/Workspace/moyi/results/moyi_top10_rf/vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocabs = json.load(f)
callig_vocab = vocabs["callig_vocab"]
script_vocab = vocabs["script_vocab"]
char_vocab = vocabs["char_vocab"]

# 2. Load eval200_fixed.csv
csv_p = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
with open(csv_p, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} rows from {csv_p}")

# 3. Load moyi_4ch Model (step 80,000)
ckpt_path = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch/checkpoints/moyi_0080000.pt"
print(f"Loading moyi_4ch from {ckpt_path}...")
model = DiT_models["moyun-4channel-B"](
    input_size=32,
    num_classes=6000,
    learn_sigma=False,
    if_rope=False
).to(device)

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt.get("ema", ckpt.get("model"))
model.load_state_dict(state_dict)
model.eval()

# 4. Load VAE
vae_path = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(vae_path).to(device)
vae.eval()

out_dir = "/home/ds/Workspace/moyi/results/moyi_top10_rf_4ch/eval_ours200fix"
os.makedirs(out_dir, exist_ok=True)

# 5. Batch Inference (batch_size = 16)
batch_size = 16
steps = 50
t0 = time.time()

for b_start in range(0, len(rows), batch_size):
    b_rows = rows[b_start:b_start+batch_size]
    B = len(b_rows)
    
    y_list = []
    for r in b_rows:
        cid = callig_vocab.get(r["calligrapher"], 0)
        sid = script_vocab.get(r["script"], 0)
        chid = char_vocab.get(r["character"], 0)
        y_list.append([cid, sid, chid])
        
    y = torch.tensor(y_list, device=device, dtype=torch.long)
    stroke = torch.zeros(B, dtype=torch.long, device=device)
    
    # 4 channels latent
    torch.manual_seed(42 + b_start)
    z = torch.randn(B, 4, 32, 32, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    
    with torch.no_grad():
        for i in range(steps):
            t = ts[i].expand(B)
            dt = ts[i + 1] - ts[i]
            v_pred = model(z, t, y, stroke)
            if isinstance(v_pred, (tuple, list)):
                v_pred = v_pred[0]
            z = z + v_pred * dt
            
        decoded = vae.decode(z / 0.18215).sample
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        
        for i_in_b, r in enumerate(b_rows):
            global_idx = b_start + i_in_b
            img_t = decoded[i_in_b]
            save_path = os.path.join(out_dir, f"g{global_idx}.png")
            save_image(img_t, save_path)
            
    print(f"Generated {min(b_start+batch_size, len(rows))}/{len(rows)} samples...")

print(f"✓ All 187 samples generated in {time.time()-t0:.2f}s!")
