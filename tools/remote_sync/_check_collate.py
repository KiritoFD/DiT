# -*- coding: utf-8 -*-
"""快速验证: 标准字形 latent npy shape + kailishu dataset batch collate 是否一致。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

# 1) npy shape 抽样
for f in ["std_glyph_latent/kai/U+0342D.npy", "std_glyph_latent/li/U+04E00.npy"]:
    a = np.load(f); print(f"{f}: shape={a.shape} dtype={a.dtype}")

# 2) batch collate 一致性
from torch.utils.data import DataLoader
from latent_dataset import MCCDLatentDataset
ds = MCCDLatentDataset(csv_file="kailishu_train.csv", latent_shards_dir="final_latents",
                       img_root="final_imgs_256", image_size=256,
                       load_canny=False, load_skel=False, load_image=True,
                       preload=False, structure_size=256, use_glyph_cond=True)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
b = next(iter(loader))
g = b['g']
print(f"batch g shape: {tuple(g.shape)} dtype={g.dtype}")
assert tuple(g.shape)[0] == 8, "g should collate to (B,4,32,32)"
assert tuple(g.shape)[1:] == (4,32,32)
print("COLLATE OK")
