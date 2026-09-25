#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加性残差是不是在"作弊"？—— 测拓扑与墨量，而不是测 MSE。

## 为什么要测这个
我的"闭合率" = 1 − MSE(g′,g_gt)/MSE(g_std,g_gt)，**无法区分**：
  (a) 形变对了（坐标挪动，把笔画搬到正确位置）
  (b) 凭空补像素（加性残差直接"画"出目标，甚至留下鬼影）
两者都能降低 MSE -> 指标会奖励作弊。

## 三个能区分它们的量
  ① **墨量比** |ink(g′)| / |ink(g_std)| —— 纯坐标形变大致守恒（1.0 附近）；
     残差若"画"东西会明显偏离。对照 |ink(g_gt)|/|ink(g_std)| 看目标本身该是多少。
  ② **连通分量数** —— 残差若留下鬼影/断笔，分量数会暴涨。
  ③ **重骨架化后的 IoU** —— 把 g′ 解码成图再骨架化，与 g_gt 比；
     纯形变应保持"细线"，残差可能变粗/多出孤立点。

用法: python tools/diag_deform_cheat.py --ckpts a.pt,b.pt
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpts', default='assets/deform_skel_standalone.pt,assets/deform_skel_v5.pt')
ap.add_argument('--std-dir', default='data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--n', type=int, default=400, help='测多少条')
ap.add_argument('--batch', type=int, default=256)
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


A, B = load_bank(a.std_dir), load_bank(a.gt_dir)
rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
C2I = {}
for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
meta = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    if i in A and i in B:
        meta[i] = C2I.get(str(r.get('calligrapher', '')), 0)
ids = sorted(meta)[:a.n]
G = torch.from_numpy(np.stack([A[i] for i in ids])).to(DEV)
T = torch.from_numpy(np.stack([B[i] for i in ids])).to(DEV)
Y = torch.tensor([meta[i] for i in ids], device=DEV)
tab = torch.load('assets/callig_emb_pretrained_50k.pt', map_location='cpu',
                 weights_only=False)['embedding'].float().to(DEV)
print(f'[1] 测 {len(ids)} 条', flush=True)

# VAE 解码（把 latent 变成可算墨量/连通性的图）
from src.eval.inference import load_eval_vae  # noqa: E402
vae = load_eval_vae(DEV)
scaling = float(getattr(vae.config, 'scaling_factor', 0.18215))


@torch.no_grad()
def decode(L):
    out = []
    for s in range(0, len(L), a.batch):
        z = L[s:s + a.batch].to(DEV) / scaling
        im = vae.decode(z).sample
        out.append(((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu().numpy())   # 灰度
    return np.concatenate(out)


def stats(img):
    """返回 (墨量, 连通分量数)。墨量=暗像素占比; 阈值取 0.5。"""
    ink = img < 0.5
    n = ink.sum()
    if n == 0:
        return 0.0, 0
    lab, k = ndimage.label(ink)
    return float(n) / img.size, int(k)


ink_std = decode(G)
ink_gt = decode(T)
print('[2] 解码完成（std / gt）', flush=True)
s_std = np.array([stats(x)[0] for x in ink_std])
s_gt = np.array([stats(x)[0] for x in ink_gt])
c_std = np.array([stats(x)[1] for x in ink_std])
c_gt = np.array([stats(x)[1] for x in ink_gt])
print(f'\n  基线(g_std): 墨量 {s_std.mean():.4f}  连通分量 {c_std.mean():.1f}')
print(f'  目标(g_gt) : 墨量 {s_gt.mean():.4f}  连通分量 {c_gt.mean():.1f}')
print(f'  -> 目标/输入 墨量比 = {s_gt.mean()/max(s_std.mean(),1e-9):.3f}  '
      f'（纯坐标形变应接近这个值）')

print(f'\n{"ckpt":>34} | {"闭合率":>7} {"墨量":>7} {"墨量比":>7} {"分量数":>7} {"MSE":>8}')
print('-' * 82)
for ck in a.ckpts.split(','):
    ck = ck.strip()
    if not os.path.exists(ck):
        print(f'{ck:>34} | 不存在'); continue
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    sd = sd.get('deform', sd)
    # 从权重形状推断 residual / width
    has_res = any('res.weight' in k for k in sd)
    wid = sd['d1.0.weight'].shape[0] if 'd1.0.weight' in sd else 96
    m = DeformSkel(cond_dim=128, ch=4, grid=32, residual=int(has_res),
                   width=int(wid), max_off=6.0, res_cap=2.0).to(DEV).eval()
    try:
        m.load_state_dict(sd, strict=True)
    except RuntimeError as _e:
        # 旧代次的 ckpt(如 v2/v3 的 style_off 是 8x8) 与现在的模块(全分辨率 32x32)形状不符。
        # 这不是 bug, 是架构演进 —— 直接跳过并说明, 别让它把整轮诊断带崩。
        print(f'{os.path.basename(ck):>34} | 跳过: 架构代次不符 ({str(_e).splitlines()[1][:60] if chr(10) in str(_e) else str(_e)[:60]})')
        continue
    with torch.no_grad():
        out = []
        for s in range(0, len(G), a.batch):
            out.append(m(G[s:s + a.batch], tab[Y[s:s + a.batch]]))
        P = torch.cat(out)
    mse = float((P - T).pow(2).mean())
    clo = 100 * (1 - mse / 0.50494)
    dec = decode(P)
    s_p = np.array([stats(x)[0] for x in dec])
    c_p = np.array([stats(x)[1] for x in dec])
    tag = os.path.basename(ck) + (' [残差]' if has_res else ' [纯形变]')
    print(f'{tag:>34} | {clo:6.1f}% {s_p.mean():7.4f} '
          f'{s_p.mean()/max(s_std.mean(),1e-9):7.3f} {c_p.mean():7.1f} {mse:8.5f}',
          flush=True)

print()
print('读法:')
print('  · 纯坐标形变: 墨量比应 ≈ 目标/输入 的比值, 分量数不应比 g_std 明显增多')
print('  · 若某版本 墨量比 明显偏离 且 分量数 明显增大 -> 它在**凭空补像素/留鬼影**,')
print('    闭合率的提升是作弊来的, 喂给 DiT 会污染条件。')
