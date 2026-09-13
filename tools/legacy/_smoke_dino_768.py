# -*- coding: utf-8 -*-
"""Smoke test: DiT-2Cond-S/4 + factorized_add with char_embed_dim=768 (DINO 768 -> 384 direct)."""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DiT_2Cond_models

torch.manual_seed(0)

# ---- build exactly like train.py (2cond branch) ----
model = DiT_2Cond_models["DiT-2Cond-S/4"](
    input_size=256 // 4,  # latent 64x64
    num_calligraphers=1011,
    num_characters=35130,
    use_checkpoint=False,
    condition_fusion="factorized_add",
    callig_embed_dim=128,
    char_embed_dim=768,
    in_channels=3,  # latent_channels=3 (kl-f4)
    cond_drop_all_prob=0.05,
    cond_drop_one_prob=0.25,
)
print("callig table:", tuple(model.y_callig_embedder.embedding_table.weight.shape))
print("char  table:", tuple(model.y_char_embedder.embedding_table.weight.shape))
cp = model.char_proj
print("char_proj: LayerNorm", cp[0].normalized_shape, "-> Linear", cp[1].in_features, "->", cp[1].out_features)
print("hidden (x_embedder proj out):", model.x_embedder.proj.out_channels)

# ---- simulate DINO injection (like train.py) ----
NUM_CH = 7026
emb = np.random.randn(20468, 768).astype(np.float32)
emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
glyphs = []
for i in range(20468):
    # 5 scripts x 7026 chars: spread synthetic glyphs across all scripts
    sid = (i * 3) % 5
    cid = (i // 5 if (i // 5) < 7026 else i % 7026)
    glyphs.append([sid, cid])
gids = [int(s) * NUM_CH + int(c) for s, c in glyphs]
assert max(gids) < 35130, f"gid overflow: {max(gids)}"

table = model.y_char_embedder.embedding_table.weight
loaded = 0
with torch.no_grad():
    for i, (s, c) in enumerate(glyphs):
        gid = s * NUM_CH + c
        table[gid].copy_(torch.from_numpy(emb[i]).float())
        loaded += 1
print(f"injected {loaded} rows")

# verify injected rows carry exact DINO vectors
with torch.no_grad():
    wrong = 0
    for i, (s, c) in enumerate(glyphs[:5000]):
        gid = s * NUM_CH + c
        if not torch.allclose(table[gid], torch.from_numpy(emb[i])):
            wrong += 1
    print(f"checked 5000 injected rows, mismatches={wrong}")
    print(f"injected row norm={table[glyphs[0][0]*NUM_CH+glyphs[0][1]].norm().item():.4f}, "
          f"random row norm={table[35130].norm().item():.4f} (CFG row, untouched)")

# ---- forward through char_proj ----
x = table[0:4]
y = cp(x)
print("char_proj out shape:", tuple(y.shape))

# ---- full model forward with a tiny batch, latent 64x64 ----
z = torch.randn(2, 3, 64, 64)  # latent_channels=3, latent 64x64
t = torch.randint(0, 1000, (2,))
y_c = torch.randint(0, 1011, (2,))
y_ch = torch.tensor([7026, 35129])  # first & last valid glyph ids
out = model(z, t, y_c, y_ch)
print("model out:", tuple(out.shape), "| dtype:", out.dtype)
print("SMOKE OK")