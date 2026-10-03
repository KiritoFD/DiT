#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线构建「骨架条件增强」的变体 shards。

原则（用户给的）：**构建期保证信号不丢失，前向期破坏其精确性**。
所以几何类扰动（弹性/仿射/区域丢弃）在像素域离线做，做完再 VAE 编码；
数值类扰动（高斯加噪 / latent patch drop / 整图 dropout）留在训练前向期（已在 train.py）。

变体：
  el1  弹性 α=10 σ=3      （弱：笔画轻微波动）
  el2  弹性 α=20 σ=4      （中：结体明显波动，笔画仍连通）
  aff  仿射 rot±4° scale 0.96~1.04 shear±2.5° trans±4px
  drp  区域丢弃 1 块 40~64px（模拟局部笔画缺失，强制补全）

产物：data/50k/shards_std_aug_<variant>/shard_XXXXX.npz  （键与 shards_std_fixed 同规则:
      按训练 csv 每行的 img_id 存，源图用该行的 std_path）

⚠ 编码必须与既有链路一致：ToTensor + Normalize([0.5]*3) + vae.encode * 0.18215
  （见 tools/rebuild_one_shard.py）。这里用 latent_dist.mode()（确定性、不额外引入
  采样噪声），并会打印 mode vs sample 的差 vs 几何扰动的差，供判断量级。
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '/root/Workspace/xy/DiT')

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='assets/train_50k_v2_fixed.csv')
ap.add_argument('--out-root', default='data/50k')
ap.add_argument('--shard-size', type=int, default=10000)
ap.add_argument('--vae-batch', type=int, default=64)
ap.add_argument('--workers', type=int, default=32)
ap.add_argument('--variants', default='el1,el2,aff,drp')
ap.add_argument('--limit', type=int, default=0, help='>0 只做前 N 条（调试）')
ap.add_argument('--preview-n', type=int, default=8)
a = ap.parse_args()


# ---------------------------------------------------------------- 扰动算子
def _elastic(b, alpha, sigma, seed):
    rng = np.random.default_rng(seed)
    f = b.astype(np.float32)
    h, w = f.shape
    dx = gaussian_filter((rng.random((h, w)) - 0.5) * 2, sigma) * alpha
    dy = gaussian_filter((rng.random((h, w)) - 0.5) * 2, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    idx = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    return map_coordinates(f, idx, order=1, mode='reflect') > 0.5


def _affine(b, seed):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-4, 4)
    sc = rng.uniform(0.96, 1.04)
    sh = np.tan(np.deg2rad(rng.uniform(-2.5, 2.5)))
    tx, ty = rng.uniform(-4, 4, 2)
    f = b.astype(np.float32)
    h, w = f.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    cy, cx = h / 2, w / 2
    th = np.deg2rad(ang)
    x0, y0 = xx - cx, yy - cy
    xr = x0 * np.cos(th) - y0 * np.sin(th)
    yr = x0 * np.sin(th) + y0 * np.cos(th)
    xr = xr - sh * yr
    xs = xr / sc + cx - tx
    ys = yr / sc + cy - ty
    return map_coordinates(f, [np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)],
                           order=1, mode='reflect') > 0.5


def _drop(b, seed):
    rng = np.random.default_rng(seed)
    out = b.copy()
    h, w = b.shape
    s = int(rng.integers(40, 65))
    y = int(rng.integers(0, max(1, h - s)))
    x = int(rng.integers(0, max(1, w - s)))
    out[y:y + s, x:x + s] = False
    return out


VARIANTS = {
    'el1': lambda b, s: _elastic(b, 10, 3, s),
    'el2': lambda b, s: _elastic(b, 20, 4, s),
    'aff': lambda b, s: _affine(b, s),
    'drp': lambda b, s: _drop(b, s),
}
WANT = [v.strip() for v in a.variants.split(',') if v.strip()]
for v in WANT:
    assert v in VARIANTS, f'未知变体 {v}'


def make_one(job):
    """worker: 读 std png -> 扰动 -> 存 PNG（供预览）-> 返回路径"""
    variant, iid, std_path, png_out = job
    try:
        img = Image.open(std_path).convert('L')
    except Exception as e:
        return (iid, None, f'{type(e).__name__}: {e}')
    ink = np.asarray(img) < 127
    out = VARIANTS[variant](ink, iid * 7919 + hash(variant) % 100003)
    arr = np.where(out, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(png_out)
    return (iid, png_out, None)


def main():
    rows = list(csv.DictReader(open(a.csv, encoding='utf-8')))
    if a.limit:
        rows = rows[:a.limit]
    sys.path.insert(0, '.')
    from src.utils.latent_dataset import extract_img_id
    items = []
    for r in rows:
        try:
            iid = extract_img_id(r, where='aug-build')
        except Exception:
            continue
        items.append((iid, r.get('std_path', '')))
    print(f'[csv] {len(items)} 条', flush=True)

    import torch
    from diffusers.models import AutoencoderKL
    from torchvision import transforms
    dev = 'cuda'
    vae = AutoencoderKL.from_pretrained(
        'data/pretrained/pretrained_models/sd-vae-ft-ema',
        local_files_only=True).to(dev).eval()
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    print('[vae] 就绪', flush=True)

    for variant in WANT:
        t0 = time.time()
        png_dir = f'{a.out_root}/std_aug_{variant}'
        out_dir = f'{a.out_root}/shards_std_aug_{variant}'
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(variant, iid, sp, f'{png_dir}/{iid:06d}.png') for iid, sp in items]
        print(f'[{variant}] 生成 {len(jobs)} 张扰动图 ...', flush=True)
        with Pool(a.workers) as pool:
            res = pool.map(make_one, jobs, chunksize=256)
        bad = [r for r in res if r[2]]
        if bad:
            print(f'[{variant}] ⚠ {len(bad)} 张失败，例: {bad[:2]}', flush=True)

        # 编码
        print(f'[{variant}] 编码 ...', flush=True)
        ids_buf, lat_buf, shard_i, total = [], [], 0, 0

        def flush(shard_i):
            if not ids_buf:
                return shard_i
            L = np.stack(lat_buf).astype(np.float16)
            np.savez_compressed(f'{out_dir}/shard_{shard_i:05d}.npz',
                                latents=L, img_ids=np.array(ids_buf, dtype=np.int64))
            print(f'    shard_{shard_i:05d}.npz  {len(ids_buf)} 条', flush=True)
            ids_buf.clear(); lat_buf.clear()
            return shard_i + 1

        buf_imgs = []
        for k, (iid, sp) in enumerate(items):
            p = f'{png_dir}/{iid:06d}.png'
            if not os.path.exists(p):
                continue
            buf_imgs.append((iid, tf(Image.open(p).convert('RGB'))))
            if len(buf_imgs) >= a.vae_batch or k == len(items) - 1:
                x = torch.stack([b for _, b in buf_imgs]).to(dev)
                with torch.no_grad():
                    lat = vae.encode(x).latent_dist.mode() * 0.18215
                lat = lat.cpu().float().numpy()
                for j, (iid2, _) in enumerate(buf_imgs):
                    ids_buf.append(iid2); lat_buf.append(lat[j])
                    total += 1
                buf_imgs = []
                if len(ids_buf) >= a.shard_size:
                    shard_i = flush(shard_i)
        shard_i = flush(shard_i)
        print(f'[{variant}] 完成 {total} 条, {shard_i} 个 shard, {time.time()-t0:.1f}s', flush=True)

    # 量级诊断: mode vs sample 的差  vs  几何扰动的差
    print('\n[诊断] mode/sample 采样噪声 vs 几何扰动量级', flush=True)
    probe = [it for it in items[:a.preview_n]]
    x0 = torch.stack([tf(Image.open(f'{a.out_root}/std_aug_{WANT[0]}/{i:06d}.png')
                         .convert('RGB')) for i, _ in probe]).to(dev)
    xb = torch.stack([tf(Image.open(sp).convert('RGB')) for _, sp in probe]).to(dev)
    with torch.no_grad():
        m0 = vae.encode(x0).latent_dist.mode() * 0.18215
        mb = vae.encode(xb).latent_dist.mode() * 0.18215
        s1 = vae.encode(xb).latent_dist.sample() * 0.18215
        s2 = vae.encode(xb).latent_dist.sample() * 0.18215
    print(f'  mode(扰动) vs mode(原图) max|Δ| = {float((m0-mb).abs().max()):.4f}   <- 几何扰动')
    print(f'  sample#1   vs sample#2   max|Δ| = {float((s1-s2).abs().max()):.4f}   <- 采样噪声')
    print(f'  mode(原图) vs sample#1   max|Δ| = {float((mb-s1).abs().max()):.4f}   <- 系统差')
    print('\n全部完成', flush=True)


if __name__ == '__main__':
    main()
