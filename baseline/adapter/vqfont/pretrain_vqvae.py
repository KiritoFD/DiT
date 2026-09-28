# -*- coding: utf-8 -*-
"""VQ-VAE pretraining for VQ-Font, script port of model/VQ-VAE.ipynb.

Adapted to 256x256 (dataset resolution) and to the SAME input transform the
FFG stage uses (Normalize 0.5/0.5 -> [-1,1]); the original notebook trained on
[-0.5,0.5] while its FFG stage fed [-1,1] — we keep the two stages consistent.
Model: content_enc_builder(1,32,256) + VQ(num_embeddings=100, dim=256) + dec_builder(32,1)
so that state_dict keys line up with load_pretrain_vae_model() in train_top10.py.

Run from baseline/VQ-Font:  python ../adapter/vqfont/pretrain_vqvae.py
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.getcwd())  # VQ-Font repo
from model import content_enc_builder, dec_builder  # noqa: E402
from model.modules import weights_init  # noqa: E402

DIT = "/root/Workspace/xy/DiT"
CONTENT_DIR = os.path.join(DIT, "baseline/data/content_font/deng")
OUT_DIR = os.path.join(DIT, "baseline/VQ-Font/weight")


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, quantized, perplexity, encodings


class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._encoder = content_enc_builder(1, 32, 256)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = dec_builder(32, 1)

    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity


class ContentImgs(Dataset):
    def __init__(self, root, transform):
        self.files = sorted(os.path.join(root, f) for f in os.listdir(root)
                            if f.endswith(".png"))
        self.transform = transform

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("L"))

    def __len__(self):
        return len(self.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_embeddings", type=int, default=100)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--commitment_cost", type=float, default=0.25)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ds = ContentImgs(CONTENT_DIR, tf)
    print(f"content imgs: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, pin_memory=True, num_workers=8)

    model = Model(args.num_embeddings, args.embedding_dim, args.commitment_cost).to(dev)
    model.apply(weights_init("xavier"))
    opt = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)

    model.train()
    cyc = None
    for it in range(args.iters):
        try:
            data = next(cyc)
        except (StopIteration, TypeError):
            cyc = iter(loader)
            data = next(cyc)
        var = torch.var(data)
        data = data.to(dev)
        opt.zero_grad()
        vq_loss, recon, ppl = model(data)
        recon_error = F.mse_loss(recon, data) / (var + 1e-8)
        loss = recon_error + vq_loss
        loss.backward()
        opt.step()

        if (it + 1) % 500 == 0:
            print(f"iter {it+1}: recon {np.mean([recon_error.item()]):.4f} "
                  f"vq {vq_loss.item():.4f} ppl {ppl.item():.2f}", flush=True)
        if (it + 1) % 10000 == 0:
            torch.save(model, os.path.join(OUT_DIR, "VQ-VAE_top10_iter{}.pth".format(it + 1)))

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model, os.path.join(OUT_DIR, "VQ-VAE_top10.pth"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "VQ-VAE_top10_Parms.pth"))
    print(f"saved -> {OUT_DIR}/VQ-VAE_top10[_Parms].pth")


if __name__ == "__main__":
    main()
