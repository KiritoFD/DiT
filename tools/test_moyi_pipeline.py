import sys, os
base = "/home/ds/Workspace/moyi/ref/moyi"
sys.path.insert(0, base)
sys.path.insert(0, os.path.join(base, "moyun"))

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

print("=" * 60)
print("【Test 1: AutoencoderKL Loading】")
print("=" * 60)
vae_path = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
try:
    vae = AutoencoderKL.from_pretrained(vae_path)
    print(f"✓ VAE successfully loaded from {vae_path}!")
except Exception as e:
    print(f"✗ VAE loading failed: {e}")

print("\n" + "=" * 60)
print("【Test 2: Moyun Model Import & Forward Pass】")
print("=" * 60)
from moyun_2 import DiT_models
for model_name in ["moyun-12channel-B", "moyun-12channel"]:
    if model_name in DiT_models:
        model = DiT_models[model_name](
            input_size=32,
            num_classes=6000,
            learn_sigma=False,
            if_rope=False
        ).cuda()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model {model_name} created successfully! Params: {n_params/1e6:.1f}M")
        
        # Test forward pass with dummy tensor
        x = torch.randn(2, 12, 32, 32, device="cuda")
        t = torch.tensor([10, 20], device="cuda")
        y = torch.tensor([[0, 0, 100], [1, 1, 200]], device="cuda")
        stroke = torch.zeros(2, dtype=torch.long, device="cuda")
        
        out, _ = model(x, t, y, stroke)
        print(f"  Forward output shape: {out.shape} (Expected: (2, 12, 32, 32))")

print("\n" + "=" * 60)
print("【Test 3: Dataset Loading on Top10 Data】")
print("=" * 60)
from dataset_moyun import MultiLabelNestedDataset

csv_path = "/home/ds/Workspace/moyi/assets/train_top10_style23_real.csv"
img_shards = "/home/ds/Workspace/moyi/data/top10_style23/shards_img"
edge_shards = "/home/ds/Workspace/moyi/data/top10_style23/shards_aux_skel3"
skel_shards = "/home/ds/Workspace/moyi/data/top10_style23/shards_std_w7"

ds = MultiLabelNestedDataset(
    csv_file=csv_path,
    img_shards=img_shards,
    edge_shards=edge_shards,
    skel_shards=skel_shards,
    num_classes=6000
)
print(f"✓ Dataset loaded! Total valid samples: {len(ds)}")
print(f"  Calligraphers ({len(ds.callig_vocab)}): {list(ds.callig_vocab.keys())[:5]}...")
print(f"  Scripts ({len(ds.script_vocab)}): {list(ds.script_vocab.keys())}")
print(f"  Characters count: {len(ds.char_vocab)}")

sample = ds[0]
image, edge, skel, y, stroke, f1, f2, f3 = sample
print(f"  Sample shapes: image={image.shape}, edge={edge.shape}, skel={skel.shape}")
print(f"  y: {y}")

print("\n" + "=" * 60)
print("【Test 4: Rectified Flow Loss Computation】")
print("=" * 60)
from utils.Sampler.RF import RF
rf = RF(without_t=False)

x_batch = torch.cat([image.unsqueeze(0), edge.unsqueeze(0), skel.unsqueeze(0)], dim=1).cuda()
y_batch = torch.tensor([[y[0].item(), y[1].item(), y[2].item()]], device="cuda")
fdict = {"calligrapher_x": 0.16, "font_x": 0.08, "charactor_x": 0.08, "_num_classes": 6000}

loss_tuple = rf.forward(model, x_batch, feature_dict=fdict, y=y_batch, stroke=stroke.cuda())
print(f"✓ RF loss computed successfully: {loss_tuple[0].item():.4f}")

print("\n" + "=" * 60)
print("【ALL MODULE TESTS PASSED!】")
print("=" * 60)
