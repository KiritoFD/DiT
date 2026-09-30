#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_gprime_montage.py — 直接看各 SkelNet 输出的 g' 长什么样。

列: [g_std 输入 | v1 输出 g' | v3 输出 g' | v4 输出 g' | g_gt 目标]
行: 6 个样本
"""
import os
import sys
import glob
import csv
import numpy as np
import torch
from PIL import Image

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def main():
    from src.model.deform_skel import DeformSkel
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    sc = 0.18215

    @torch.no_grad()
    def dec(z):
        outs = []
        for i in range(0, z.shape[0], 16):
            im = vae.decode(z[i:i + 16].to(DEV).float() / sc).sample
            outs.append((((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()))
        return torch.cat(outs, 0).numpy()

    G, gid = load_all('data/top10_style23/shards_std')
    T, _ = load_all('data/top10_style23/shards_aux_skel3')
    Y = torch.load('assets/callig_script_emb_top10.pt', map_location='cpu', weights_only=False)
    if isinstance(Y, dict):
        Y = Y.get('emb', Y.get('weight', list(Y.values())[0]))
    Y = Y.float()

    rows = list(csv.DictReader(open('assets/train_top10_style23.csv', encoding='utf-8')))
    slots = sorted({r['slot_name'] for r in rows})
    s2i = {s: i for i, s in enumerate(slots)}
    id2s, id2n = {}, {}
    for r in rows:
        try:
            id2s[int(r['img_id'])] = s2i[r['slot_name']]
            id2n[int(r['img_id'])] = (r['character'], r['slot_name'])
        except Exception:
            pass
    rng = np.random.RandomState(0)
    sel = rng.choice(len(G), 6, replace=False)
    Gs = torch.from_numpy(G[sel]).to(DEV)
    Ts = torch.from_numpy(T[sel]).to(DEV)
    sid = torch.from_numpy(np.array([id2s.get(int(i), 0) for i in np.array(gid)[sel]])).to(DEV)
    Ys = Y.to(DEV)

    def build(path, width, topo, stok):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck.get('deform', ck) if isinstance(ck, dict) else ck
        m = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=width, max_off=3.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=0,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=topo,
                       warp_iters=1, film_mode='film', style_tokens=stok, attn_heads=4,
                       preserve_amp=0).to(DEV)
        cur = m.state_dict()
        keep = {k: v for k, v in sd.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
        m.load_state_dict(keep, strict=False)
        return m.eval()

    outs = {}
    with torch.no_grad():
        for tag, (p, w, tp, st) in {'v1': ('assets/deform_skel_top10_v1.pt', 96, 0, 0),
                                    'v3': ('assets/deform_skel_top10_v3_probe.pt', 128, 1, 0),
                                    'v4': ('assets/deform_skel_top10_v4_styleattn.pt', 128, 1, 16)}.items():
            m = build(p, w, tp, st)
            outs[tag] = dec(m(Gs, Ys[sid]))
            del m
            torch.cuda.empty_cache()

    col = [dec(Gs), outs['v1'], outs['v3'], outs['v4'], dec(Ts)]
    names = ['g_std 输入', "v1 g'", "v3 g'", "v4 g'", 'g_gt 目标']
    for k, n in enumerate(names):
        ink = (col[k] < 0.5).mean()
        print(f"  {n:<12} 墨量 {ink:.4f}  灰度均值 {col[k].mean():.3f}  min {col[k].min():.3f}")
    print("  样本:", [id2n.get(int(i), '?') for i in np.array(gid)[sel]])

    rows_img = []
    for r in range(6):
        rows_img.append(np.concatenate([col[k][r] for k in range(5)], axis=1))
    M = np.concatenate(rows_img, 0)
    M = (np.clip(M, 0, 1) * 255).astype(np.uint8)
    os.makedirs('/tmp/mont', exist_ok=True)
    Image.fromarray(M).save('/tmp/mont/gprime.png')
    print(f"  saved /tmp/mont/gprime.png {M.shape}  列 = {names}")


if __name__ == "__main__":
    main()
