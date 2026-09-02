# -*- coding: utf-8 -*-
"""本地 GPU 构建 skel_bank_train_1px.npz
fame 训练集每字第一张 GT 图 -> 1px 骨架 (skeletonize, 不膨胀) -> VAE latent.
与 3px 版 _diag/build_skel_banks.py 库1 逻辑一致, 仅骨架宽度不同.
"""
import os, sys, csv
import numpy as np
import torch
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
from skimage.morphology import skeletonize
from diffusers.models import AutoencoderKL


def main():
    dev = torch.device('cuda')
    vae = AutoencoderKL.from_pretrained('pretrained_models/sd-vae-ft-ema').to(dev).eval()

    rows = list(csv.DictReader(open('5script/train_fame.csv', encoding='utf-8')))
    first = {}
    for r in rows:
        key = r['script'] + '|' + r['character']
        if key not in first:
            first[key] = r['image_path']
    keys1 = sorted(first)
    print(f'[pairs] {len(keys1)} unique (script, char)', flush=True)

    lat1 = np.empty((len(keys1), 4, 32, 32), np.float16)
    B = 1
    with torch.no_grad():
        for i in range(0, len(keys1), B):
            u8 = []
            for k in keys1[i:i + B]:
                a = np.asarray(Image.open(first[k]).convert('L'))
                sk = skeletonize(a < 127)                       # 1px 骨架, 不膨胀
                u8.append(np.where(sk, 0, 255).astype('uint8'))
            x = torch.from_numpy(np.stack(u8).astype(np.float32) / 255. * 2 - 1)[:, None].repeat(1, 3, 1, 1).to(dev)
            lat1[i:i + len(u8)] = (vae.encode(x).latent_dist.mode() * 0.18215).half().cpu().numpy()
            if (i // B) % 20 == 0:
                print(f'[encode] {i}/{len(keys1)}', flush=True)

    np.savez_compressed('_sync_work/skel_bank_train_1px.npz', latents=lat1, keys=np.array(keys1))
    print('[done] skel_bank_train_1px.npz:', len(keys1), flush=True)


if __name__ == '__main__':
    main()