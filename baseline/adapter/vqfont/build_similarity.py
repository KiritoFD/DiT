# -*- coding: utf-8 -*-
"""Content-similarity matrix for VQ-Font (replaces the missing 0-byte
weight/all_char_similarity_unicode.json).

Sim(trg, ref) = cosine between flattened content-encoder features, exactly the
quantity Get_style_global looks up as chars_sim_dict[trg_uni][ref_uni].
Stored as an fp32 npy [N,N] + uni list; a dict-like SimMatrix wrapper in
train_top10.py / sample_top10.py keeps the original model code untouched.

Run from baseline/VQ-Font:  python ../adapter/vqfont/build_similarity.py
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.getcwd())
from model import content_enc_builder  # noqa: E402
from pretrain_vqvae import Model, VectorQuantizer  # noqa: E402  # unpickle needs the classes

DIT = "/root/Workspace/xy/DiT"
CONTENT_DIR = os.path.join(DIT, "baseline/data/content_font/deng")
VAE_PTH = os.path.join(DIT, "baseline/VQ-Font/weight/VQ-VAE_top10.pth")
OUT_NPY = os.path.join(DIT, "baseline/VQ-Font/weight/char_sim_top10.npy")
OUT_UNI = os.path.join(DIT, "baseline/VQ-Font/weight/char_sim_top10_unis.json")


class Imgs(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("L"))

    def __len__(self):
        return len(self.files)


def main():
    files = sorted(os.path.join(CONTENT_DIR, f) for f in os.listdir(CONTENT_DIR)
                   if f.endswith(".png"))
    unis = [f"{ord(os.path.basename(f)[:-4]):X}" for f in files]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])
    model = torch.load(VAE_PTH, map_location=dev)
    enc = model._encoder.to(dev).eval()

    feats = []
    with torch.no_grad():
        for x in DataLoader(Imgs(files, tf), batch_size=64, num_workers=8):
            z = enc(x.to(dev))                       # [B,256,32,32]
            feats.append(z.flatten(1).cpu())         # flatten -> cosine target
    F_all = torch.cat(feats)
    F_all = F_all / (F_all.norm(dim=1, keepdim=True) + 1e-8)
    sim = (F_all @ F_all.t()).numpy().astype(np.float32)   # [N,N] in [-1,1]

    np.save(OUT_NPY, sim)
    with open(OUT_UNI, "w", encoding="utf-8") as f:
        json.dump(unis, f)
    print(f"sim matrix {sim.shape} -> {OUT_NPY}")


if __name__ == "__main__":
    main()
