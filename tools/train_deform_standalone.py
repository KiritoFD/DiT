#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单独训练 DeformSkel —— 不碰扩散模型，几分钟出结论。

## 为什么能单独训
形变模块是一个小网络: (g_std, 书家风格) -> g'。
目标**现成且稠密**: g' 应逼近"该书家写的那个字"的 GT 骨架（`shards_aux_skel3`，
实测逐样本比值 0.996，宽度 ~3px 最接近 std 的 ~4px）。就是一个监督回归任务。

## 三个判据
  ① **闭合率** = 1 − MSE(g',g_gt) / MSE(g_std,g_gt)
     >60% 说明形变确实把 g_std 拉近该书家的写法
  ② **style-follow（关键）**：同一个字，对每个书家 k 生成 g'(c,k)，
     再看 g'(c,k) 是否比 g'(c,k') 更接近 **k 自己的** g_gt(c,k)。
     这一项直接回答"同一个 skel 是否对不同书家产出对应 skel"。
  ③ **offset 幅值**（含风格底图那一份）: ≈0 = 退化成恒等

## ★ v2 的两处修正（v1 的问题）
  · batch 太小(512) -> 改成吃满显存
  · **同字分组采样**：每个 batch 由若干"字"组成，每个字取多个书家 ->
    同一个 g_std 在同批里对应多个不同的 g_gt，**逼模型必须用风格去区分**。
    v1 用随机采样时同批几乎不会出现同字，风格因此被忽略（correct≈shuffled）。

## 用法
  python tools/train_deform_standalone.py --steps 6000 --batch 4096 --group 32
产出 assets/deform_skel_standalone.pt
"""
import argparse
import collections
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/root/Workspace/xy/DiT')
sys.path.insert(0, '.')
from src.model.deform_skel import DeformSkel  # noqa: E402

# ★ 关掉 cuDNN TF32。本机实测: conv 走 NHWC 的 TF32 张量核之前，cuDNN 必须先做
#   NCHW->NHWC 布局转换（`nchwToNhwcKernel`，算子级 profiler 里占 13% CUDA 时间、
#   700 次调用/10 步）。关掉 TF32 后直接用 fp32 核，省掉这批转换，整体更快。
torch.backends.cudnn.allow_tf32 = False

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--batch', type=int, default=4096)
ap.add_argument('--group', type=int, default=32, help='每个 batch 采多少个"字"')
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--std-dir', default='data/50k/shards_std_fixed')
ap.add_argument('--gt-dir', default='data/50k/shards_aux_skel3')
ap.add_argument('--out', default='assets/deform_skel_standalone.pt')
ap.add_argument('--init', default='', help='从已有头的权重续训(不换网络)')
ap.add_argument('--blur', type=int, default=0, help='输入拼一路高斯模糊(默认关)')
ap.add_argument('--ckpt', type=int, default=0, help='梯度检查点(默认关; 用朴素 batch 4096)')
ap.add_argument('--dt-ch', type=int, default=1, dest='dt_ch',
                help='距离场输入通道数(默认 1). 对 g_std 是固定量, 离线预计算一次')
ap.add_argument('--blur-sigma', type=float, default=1.5, dest='blur_sigma')
ap.add_argument('--w-tv-out', type=float, default=1e-2, dest='w_tv_out',
                help='latent 输出梯度能量惩罚(压破碎, 不需要 VAE)')
ap.add_argument('--w-tv-res', type=float, default=1e-2, dest='w_tv_res',
                help='残差梯度能量惩罚(残差是凭空写像素的唯一入口)')
ap.add_argument('--w-tv', type=float, default=1e-2, dest='w_tv',
                help='偏移场 TV 平滑正则(防把横线扭成波浪)')
ap.add_argument('--w-fold', type=float, default=1e-1, dest='w_fold',
                help='Jacobian 折叠惩罚 det(I+∇U)>0(防空间折叠/笔画翻转)')
ap.add_argument('--diag-decode', type=int, default=1, dest='diag_decode',
                help='评估时解码看墨量/破碎(只诊断, 不进训练). 1=开')
ap.add_argument('--w-img', type=float, default=0.0, dest='w_img',
                help='图像空间监督权重: 解码 g-prime 后与 g_gt 的骨架比结构。这是关键项 —— 实测 latent 余弦 0.947 但解码后碎成 28 段，latent MSE 控制不住连通性')
ap.add_argument('--img-batch', type=int, default=128, dest='img_batch',
                help='图像监督每步解码多少条(全 batch 解码太贵)')
ap.add_argument('--vae', default='data/pretrained/pretrained_models/sd-vae-ft-ema')
ap.add_argument('--lr-min-ratio', type=float, default=0.1, dest='lr_min_ratio')
ap.add_argument('--eval-every', type=int, default=1000)
ap.add_argument('--residual', type=int, default=0, help='1=形变+加性残差')
ap.add_argument('--style-dim', type=int, default=128, dest='style_dim')
ap.add_argument('--style-emb', default='assets/callig_emb_pretrained_50k.pt',
                dest='style_emb',
                help='用**模型同一张**预训练书家表(冻结); 否则离线训的风格输入与'
                     '线上 _e_callig() 对不上, 训好的头接进去会失效')
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--max-off', type=float, default=3.0, dest='max_off')
ap.add_argument('--res-cap', type=float, default=1.0, dest='res_cap')
ap.add_argument('--follow-n', type=int, default=400, help='style-follow 测多少个字')
a = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)
NCAL = 45


def load_bank(d):
    m = {}
    for f in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        z = np.load(f)
        L, I = z['latents'], z['img_ids']
        for j, i in enumerate(I):
            m[int(i)] = L[j].astype(np.float32)
        z.close()
    return m


print('[1] 载入 g_std 与 g_gt ...', flush=True)
A = load_bank(a.std_dir)
B = load_bank(a.gt_dir)
print(f'    g_std {len(A)} / g_gt {len(B)}', flush=True)

C2I = {}
rows = list(csv.DictReader(open('assets/train_50k_v2_fixed.csv', encoding='utf-8')))
for r in rows:
    C2I.setdefault(str(r.get('calligrapher', '')), len(C2I))
rec = {}
for r in rows:
    try:
        i = int(os.path.basename(r.get('image_path', '')).split('.')[0])
    except Exception:
        continue
    if i in A and i in B:
        rec[i] = (C2I.get(str(r.get('calligrapher', '')), 0), str(r.get('character', '')))
ids = sorted(rec)
print(f'    有效 {len(ids)} 条, 书家 {len(C2I)} 个', flush=True)

G = torch.from_numpy(np.stack([A[i] for i in ids])).to(DEV)
T = torch.from_numpy(np.stack([B[i] for i in ids])).to(DEV)
Y = torch.tensor([rec[i][0] for i in ids], device=DEV)
CH = [rec[i][1] for i in ids]
N = len(ids)

# 字 -> 该字的样本下标（用于分组采样）
by_char = collections.defaultdict(list)
for k, c in enumerate(CH):
    by_char[c].append(k)
# 只保留"至少 2 个不同书家"的字（style-follow 与分组采样都需要）
groups = [v for v in by_char.values()
          if len({int(Y[i]) for i in v}) >= 2]
print(f'[1] 可用于分组/跟随测试的字 {len(groups)} 个', flush=True)
# ★ 向量化分组采样: 把 groups 拼成 (G, K) 的 int64 数组 + 掩码, 每步只用 numpy 花式索引。
#   原来每步跑 48 次 Python 循环(每次 numpy 调用), 是 step 里最大的 CPU 开销。
_K = max(len(v) for v in groups)
_GT = np.full((len(groups), _K), -1, np.int64)
_GV = np.zeros((len(groups), _K), bool)
for _gi, _v in enumerate(groups):
    _GT[_gi, :len(_v)] = _v
    _GV[_gi, :len(_v)] = True
print(f'[1] 分组表 {_GT.shape} (K={_K})', flush=True)

# ── 距离场输入（infra: 对 g_std 是固定量 -> 只算一次）──────────────────
DT = None
if a.dt_ch > 0:
    from scipy import ndimage
    _t0 = time.time()
    _mag = np.abs(G.cpu().numpy()).mean(1)               # (N,32,32) 幅度图
    _thr = np.median(_mag)
    _dt = np.empty_like(_mag)
    for _k in range(len(_mag)):
        _dt[_k] = ndimage.distance_transform_edt(_mag[_k] <= _thr)
    _dt = _dt / max(_dt.max(), 1e-6)
    DT = torch.from_numpy(_dt[:, None]).to(DEV)          # (N,1,32,32)
    print(f'[1b] 距离场预计算完成 {tuple(DT.shape)} {time.time()-_t0:.0f}s '
          f'(阈值 {_thr:.3f}, 前景占比 {float((_mag>_thr).mean()):.3f})', flush=True)

base_mse = float((G - T).pow(2).mean())
print(f'\n[2] 基线 MSE(g_std, g_gt) = {base_mse:.5f}   <- 形变要打败的就是它', flush=True)

model = DeformSkel(cond_dim=a.style_dim, ch=4, grid=32, residual=a.residual,
                   width=a.width, max_off=a.max_off, res_cap=a.res_cap,
                   blur=a.blur, blur_sigma=a.blur_sigma,
                   dt_ch=a.dt_ch, affine=1, ckpt=a.ckpt).to(DEV)
if a.init:
    _sd = torch.load(a.init, map_location='cpu', weights_only=False)
    _sd = _sd.get('deform', _sd)
    # ⚠ 加了 blur 输入通道后 d1 的 in_channels 变了 -> 形状不符的键会被跳过。
    #   注意 load_state_dict(strict=False) **仍然会因形状不符报错**(strict=False 只忽略
    #   缺失/多余键, 不忽略形状), 所以必须自己先过滤。
    _cur = model.state_dict()
    _keep = {k: v for k, v in _sd.items()
             if k in _cur and tuple(_cur[k].shape) == tuple(v.shape)}
    _skip = [k for k in _sd if k not in _keep]
    model.load_state_dict(_keep, strict=False)
    print(f'[2] 续训: 从 {a.init} 载入 {len(_keep)}/{len(_sd)} 个键'
          f'（跳过形状不符 {len(_skip)} 个: {_skip[:3]}）', flush=True)
# ★ 风格源必须与模型 _e_callig() 一致: 那就是 callig_emb_pretrained_50k.pt 的行。
#   用自己学的 embedding 会导致"离线训好的头接进模型后风格输入分布不一致"而失效。
_emb = torch.load(a.style_emb, map_location='cpu', weights_only=False)
_tab = _emb['embedding'] if isinstance(_emb, dict) else _emb
_tab = _tab.float()
assert _tab.shape[0] == NCAL, _tab.shape
print(f'[2] 用预训练书家表 {a.style_emb} {tuple(_tab.shape)} (冻结)', flush=True)
style = nn.Embedding.from_pretrained(_tab, freeze=True).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
print(f'[2] 形变模块参数 {sum(p.numel() for p in model.parameters()):,}  '
      f'(含风格底图 {sum(p.numel() for p in model.style_off.parameters()):,})', flush=True)

n_val = max(1, int(N * 0.03))
perm = np.random.permutation(N)
val_i = torch.tensor(perm[:n_val], device=DEV)
tr_i = torch.tensor(perm[n_val:], device=DEV)

# style-follow 评测集：每个字取最多 6 个不同书家的样本
follow = []
for v in groups:
    byk = {}
    for i in v:
        byk.setdefault(int(Y[i]), i)
    if len(byk) >= 2:
        follow.append(list(byk.values())[:6])
    if len(follow) >= a.follow_n:
        break
print(f'[2] style-follow 用 {len(follow)} 个字', flush=True)


@torch.no_grad()
def style_follow():
    """同一个字、不同书家: g'(c,k) 是否比 g'(c,k') 更接近 k 自己的 g_gt(c,k)。"""
    model.eval()
    good = tot = 0
    same_g = []
    for v in follow:
        gs = G[v]                                  # 同字的多个样本（g_std 基本同图）
        same_g.append(float((gs - gs[0]).abs().mean()))
        for i in v:
            others = [j for j in v if int(Y[j]) != int(Y[i])]
            if not others:
                continue
            _d = DT[i:i + 1] if DT is not None else None
            gp_self = model(G[i:i + 1], style(Y[i:i + 1]), dt=_d)[0]
            gp_oth = model(G[i:i + 1], style(Y[others[0]:others[0] + 1]), dt=_d)[0]
            d_self = float((gp_self - T[i]).pow(2).mean())
            d_oth = float((gp_oth - T[i]).pow(2).mean())
            good += int(d_self < d_oth)
            tot += 1
    model.train()
    return (good / tot if tot else float('nan')), (np.mean(same_g) if same_g else float('nan'))


@torch.no_grad()
def evaluate(tag):
    model.eval()
    vi = val_i
    g2 = model(G[vi], style(Y[vi]), dt=(DT[vi] if DT is not None else None))
    mse = float((g2 - T[vi]).pow(2).mean())
    ysh = (Y[vi] + 3) % NCAL
    g2s = model(G[vi], style(ysh), dt=(DT[vi] if DT is not None else None))
    off = model.offset_stats() or {}
    fr, sg = style_follow()
    # 不可约下界(同字同书家组内方差) —— 判"收敛没收敛"要看残差/下界, 而不是看残差绝对值。
    # 实测下界 = 0.06505 (tools/diag_skelnet_floor.py), 闭合率天花板 = 87.1%。
    _FLOOR = 0.06505
    print('  %-10s MSE=%.5f (基线 %.5f, 闭合 %5.1f%%) | correct %.5f / shuffled %.5f '
          '| 输出变化 %.4f | off %.4f (风格底图 %.4f) | **style-follow %.1f%%**'
          % (tag, mse, base_mse, 100 * (1 - mse / base_mse),
             float((g2 - T[vi]).pow(2).mean()), float((g2s - T[vi]).pow(2).mean()),
             float((g2 - g2s).abs().mean()), off.get('mean_abs', 0),
             off.get('style_part', 0), 100 * fr), flush=True)
    # 判"收敛没收敛"要看 残差/下界，而不是残差绝对值。
    # 下界 = 0.06505（同字同书家组内方差，tools/diag_skelnet_floor.py 实测），天花板 = 87.1%。
    _ex = ''
    if vae is not None and a.diag_decode > 0:   # decode 只在评估里当诊断, 不进训练
        with torch.no_grad():
            gp = model(G[vi[:64]], style(Y[vi[:64]]),
                       dt=(DT[vi[:64]] if DT is not None else None))
            pr = _decode_gray(gp)
            tg = _decode_gray(T[vi[:64]])
            ink_p = float((pr < 0.5).float().mean())
            ink_t = float((tg < 0.5).float().mean())
            _ex = ' | **图空间**: 墨量 %.4f / 目标 %.4f (比 %.2f)' % (
                ink_p, ink_t, ink_p / max(ink_t, 1e-9))
    print('             -> 残差/下界 = %.2fx | 闭合 %.1f%% / 天花板 87.1%%%s'
          % (mse / 0.06505, 100 * (1 - mse / base_mse), _ex), flush=True)
    model.train()
    return mse


# ── 图像空间监督: 需要 VAE ─────────────────────────────────────────
vae = None
_vae_on_gpu = False
_sc = 0.18215
if a.w_img > 0 or a.diag_decode > 0:
    from diffusers.models import AutoencoderKL
    # ★ VAE 只在 eval 诊断时用一次 -> 常驻 CPU, 解码前再搬上 GPU(省 ~1.5G 常驻显存)
    #   但 w_img>0 时每步都要解码 -> 常驻 GPU, 否则每步来回 PCIe 搬 160MB 太慢。
    vae = AutoencoderKL.from_pretrained(a.vae, local_files_only=True).eval()
    for _p in vae.parameters():
        _p.requires_grad_(False)
    _sc = float(getattr(vae.config, 'scaling_factor', 0.18215))
    _vae_on_gpu = bool(a.w_img > 0)
    if _vae_on_gpu:
        vae.to(DEV)
    print('[2b] VAE 就绪 (scaling=%.5f, 常驻 %s)' % (_sc, 'GPU' if _vae_on_gpu else 'CPU'),
          flush=True)
    print('[2b] 目标图**不预解码**: 全量 50786 张 256x256 float32 = 13.3GB, 会 OOM。'
          '改成每步只解码当前 batch 的 %d 条。' % a.img_batch, flush=True)


def _decode_gray(lat, grad=False):
    """latent -> 灰度图 (n,256,256) in [0,1]。

    ⚠ 训练用的那条路**必须**开梯度: 旧版把整个函数包了 @torch.no_grad(),
      于是 img_loss 返回的 tensor requires_grad=False -> 加进 loss 也**不回传**,
      `--w-img` 就是个静默空操作。诊断/评估那条路仍然 no_grad。
    """
    if _vae_on_gpu:
        pass
    else:
        vae.to(DEV)
    try:
        with (torch.enable_grad() if grad else torch.no_grad()):
            _im = vae.decode(lat / _sc).sample
            return ((_im.clamp(-1, 1) + 1) / 2).mean(1).float()
    finally:
        if not _vae_on_gpu:
            vae.to('cpu')
            if DEV != 'cpu':
                torch.cuda.empty_cache()


def img_loss(gp, idx):
    """解码 g-prime 与 g_gt 的**图像**比结构。

    为什么必须加: 实测 latent 余弦 0.947 / MSE 0.129 看着很好, 但解码出来墨量只剩 40%、
    连通分量从 1.7 涨到 28.4 —— latent MSE 完全控制不住图像空间的连通性。
    结构项: L1(管位置) + 边缘梯度差(管"线还在不在")。
    """
    _n = min(a.img_batch, gp.shape[0])
    _pr = _decode_gray(gp[:_n], grad=True)
    _tg = _decode_gray(T[idx[:_n]], grad=False)
    _l1 = (_pr - _tg).abs().mean()
    _gx = lambda x: (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    _gy = lambda x: (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return _l1 + (_gx(_pr) - _gx(_tg)).abs() + (_gy(_pr) - _gy(_tg)).abs()


print('\n[3] 训练 %d 步 (batch %d, 每批 %d 个字 分组采样)' % (a.steps, a.batch, a.group),
      flush=True)
evaluate('step0')
t0 = time.time()
per = max(1, a.batch // a.group)
for step in range(a.steps):
    _gi = np.random.randint(0, len(groups), a.group)          # 选 a.group 个字
    _pos = np.random.randint(0, _K, (a.group, per))           # 每字随机取 per 个位置
    _sel = _GT[_gi[:, None], _pos]                            # (a.group, per) 花式索引
    _bad = _sel < 0                                           # 落到填充位
    if _bad.any():
        _sel[_bad] = _GT[_gi[np.nonzero(_bad)[0]], 0]         # 回退到该组的第 0 个
    bi = torch.from_numpy(_sel.reshape(-1)).to(DEV)
    g2 = model(G[bi], style(Y[bi]), dt=(DT[bi] if DT is not None else None))
    loss = (g2 - T[bi]).pow(2).mean()
    # ★ TV + Jacobian: 防撕裂/防折叠（纯几何量, 作用在偏移场）
    _r = model.regularizers()
    if _r is not None:
        loss = loss + a.w_tv * _r['tv'] + a.w_fold * _r['fold']
        loss = loss + a.w_tv_out * _r.get('tv_out', 0.0) + a.w_tv_res * _r.get('tv_res', 0.0)
    # ★ 图像空间监督: latent MSE 控制不住解码后的连通性(实测墨量 40%/28 段)。
    #   w_img>0 时才解码当前 batch 的前 img_batch 条, 走带梯度的 _decode_gray。
    if a.w_img > 0 and vae is not None:
        loss = loss + a.w_img * img_loss(g2, bi)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if (step + 1) % 500 == 0:
        print('    step %5d  loss %.5f  %.0fs' % (step + 1, float(loss), time.time() - t0),
              flush=True)
    if (step + 1) % a.eval_every == 0:
        evaluate('step%d' % (step + 1))

print('\n[4] 最终', flush=True)
evaluate('final')
print()
print('=== 判读 ===')
print('  闭合率: >60% = 形变确实把 g_std 拉近该书家的写法')
print('  style-follow: 50% = 与随机无异(风格没驱动); >75% = 同一个 skel 对不同书家产出了对应 skel')
print('  off 的"风格底图"那一项: >0 说明风格专属的全局形变在起作用')
torch.save(dict(deform=model.state_dict(), style_emb=a.style_emb), a.out)
print('  已存', a.out, flush=True)
