#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_probe_loss_space.py — 「把 loss 搬到 probe 输出空间算 MSE / L1」到底可不可行?

判据只有三条 (缺一不可):
  C1. **无退化吸引子**: 空白/常数/错误骨架 的 loss 必须显著**高于**好预测。
      (原始骨架 latent 空间里这条是**破的**: MSE(空白, 目标)=0.297 < MSE(g_std, 目标)=0.487)
  C2. **单调**: loss 与真实像素骨架质量 (IoU) 同向。
  C3. **梯度方向对**: -∇loss 与「指向真值」方向一致; 且不能只是原始 loss 的一个缩放
      (若 cos(∇probe, ∇raw)≈1, 说明 probe 没提供新信息)。

同时对比 MSE 与 L1 两种取法。
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VAE = "data/pretrained/pretrained_models/sd-vae-ft-ema"
IMG_SH = "data/top10_style23/shards_img"
SKEL_SH = "data/top10_style23/shards_aux_skel3"
STD_SH = "data/top10_style23/shards_std"
PROBE = sys.argv[1] if len(sys.argv) > 1 else \
    "assets/structure_probes/latent_skel_probe_v1_mse/best.pt"
N = 32


def idx(d):
    o = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, iid in enumerate(z["img_ids"]):
                o[int(iid)] = (sp, j)
    return o


def get(m, iid):
    sp, j = m[iid]
    with np.load(sp) as z:
        return z["latents"][j].astype(np.float32)


def main():
    from diffusers.models import AutoencoderKL
    from src.train.latent_structure import LatentSkelProbe
    from skimage.morphology import skeletonize
    from scipy.ndimage import binary_dilation, generate_binary_structure

    ck = torch.load(PROBE, map_location="cpu", weights_only=False)
    pa = ck["args"]
    probe = LatentSkelProbe(pa["in_channels"], pa["out_channels"], pa["width"], pa["depth"])
    ms = probe.load_state_dict(ck["model"], strict=False)
    assert not ms.missing_keys and not ms.unexpected_keys, ms
    probe = probe.eval().to(DEV)
    for p in probe.parameters():
        p.requires_grad_(False)
    print(f"[probe] {PROBE}  args={pa}  metrics={ck.get('metrics')}")

    vae = AutoencoderKL.from_pretrained(VAE, local_files_only=True).eval().to(DEV)
    sc = float(getattr(vae.config, "scaling_factor", 0.18215))
    for p in vae.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def dec(lat):
        im = vae.decode(lat.to(DEV).float() / sc).sample
        return ((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()

    @torch.no_grad()
    def enc(x01):
        x = x01.to(DEV).float() * 2 - 1
        if x.dim() == 2:
            x = x[None, None]
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return vae.encode(x.repeat(1, 3, 1, 1)).latent_dist.mean * sc

    ii, si, di = idx(IMG_SH), idx(SKEL_SH), idx(STD_SH)
    common = sorted(set(ii) & set(si) & set(di))
    rng = np.random.RandomState(0)
    pick = [common[k] for k in rng.choice(len(common), N, replace=False)]
    print(f"[data] img {len(ii)} / skel {len(si)} / std {len(di)} -> 三方交集 {len(common)}，取 {N}")

    X = torch.from_numpy(np.stack([get(ii, i) for i in pick]))
    Y = torch.from_numpy(np.stack([get(si, i) for i in pick]))
    G = torch.from_numpy(np.stack([get(di, i) for i in pick]))
    z_bg = enc(torch.ones(1, 1, 256, 256)).float().cpu()
    zbg = z_bg.expand_as(X)

    ST = generate_binary_structure(2, 2)

    def skel_of(gray):
        """灰度图 -> 3px 骨架 bool (与 aux_skel3 同配方)。"""
        out = []
        for a in gray.cpu().numpy():
            k = skeletonize(a < 0.5)
            out.append(torch.from_numpy(binary_dilation(k, ST, iterations=1)))
        return torch.stack(out)

    def iou(a, b):
        return float((a & b).sum().float() / (a | b).sum().clamp_min(1).float())

    GT_SK = skel_of(dec(Y))          # 目标骨架(像素), 上限口径 IoU 0.956
    print(f"[data] 目标骨架墨量 {float(GT_SK.float().mean()):.4f}")

    # ══════════════════════════════════════════════════════════════════
    # C1. 退化吸引子: 好预测 / 空白 / 常数 / g_std 各 loss 排序
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96)
    print("C1. 各种「预测」的 loss 排序  —— 好的预测必须 loss 最低")
    print("=" * 96)
    meanY = Y.mean(0, keepdim=True).expand_as(Y)
    meanX = X.mean(0, keepdim=True).expand_as(X)
    cands = [
        ("好预测 (=真值 X)", X),
        ("空白 (白底 latent)", zbg),
        ("常数 (X 的均值)", meanX),
        ("g_std 标准骨架", G),
    ]
    hdr = f"  {'预测':<22}{'rawL2':>10}{'rawL1':>10}{'probeL2':>10}{'probeL1':>10}{'骨架IoU':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows_c1 = []
    for name, P in cands:
        with torch.no_grad():
            r2 = float(F.mse_loss(P, X)); r1 = float(F.l1_loss(P, X))
            p2 = float(F.mse_loss(probe(P.to(DEV)).float().cpu(), Y))
            p1 = float(F.l1_loss(probe(P.to(DEV)).float().cpu(), Y))
            iou_ = iou(skel_of(dec(P)), GT_SK)
        rows_c1.append((name, r2, r1, p2, p1, iou_))
        print(f"  {name:<22}{r2:>10.5f}{r1:>10.5f}{p2:>10.5f}{p1:>10.5f}{iou_:>10.4f}")

    print("\n  排序判读 (第 2 行是「空白」, 第 4 行是「同字标准骨架」):")
    for col, nm in [(1, "rawL2"), (2, "rawL1"), (3, "probeL2"), (4, "probeL1")]:
        v = {r[0]: r[col] for r in rows_c1}
        good = v["好预测 (=真值 X)"]
        blank = v["空白 (白底 latent)"]
        gstd = v["g_std 标准骨架"]
        ok = (good < blank) and (good < gstd)
        print(f"    {nm:<9} 好={good:.5f}  空白={blank:.5f}  常数={v['常数 (X 的均值)']:.5f}  "
              f"g_std={gstd:.5f}   -> {'✓ 排序正确' if ok else '✗ 有退化吸引子'}")

    # ══════════════════════════════════════════════════════════════════
    # C1b. ★ 换成 SkelNet 的真实目标 Y (骨架 latent) —— 这才是原空间塌陷的现场
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96)
    print("C1b. 目标换成 **骨架 latent Y** (SkelNet 的真实任务): 原始空间 vs probe 空间")
    print("=" * 96)
    cands_b = [
        ("probe(X)  [probe 空间的好预测]", probe(X.to(DEV)).float().cpu()),
        ("g_std    [无形变基线]", G),
        ("常数 mean(Y)", meanY),
        ("空白 白底 latent", z_bg.expand_as(Y)),
    ]
    hdr = f"  {'预测':<30}{'L2(·,Y)':>12}{'L1(·,Y)':>12}{'骨架IoU':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, P in cands_b:
        with torch.no_grad():
            l2 = float(F.mse_loss(P, Y)); l1 = float(F.l1_loss(P, Y))
            iou_ = iou(skel_of(dec(P)), GT_SK)
        print(f"  {name:<30}{l2:>12.5f}{l1:>12.5f}{iou_:>10.4f}")
    with torch.no_grad():
        _g2 = float(F.mse_loss(G, Y)); _b2 = float(F.mse_loss(z_bg.expand_as(Y), Y))
    print(f"\n  ★ 原始骨架 latent 空间: MSE(g_std, Y)={_g2:.5f} vs MSE(空白, Y)={_b2:.5f} "
          f"-> {'✗ 空白更优 = 退化吸引子!' if _b2 < _g2 else '✓ 空白更差, 排序正常'}")

    # ══════════════════════════════════════════════════════════════════
    # C2/C3. 沿「墨量塌陷」方向: 单调性 + 动态范围 + 梯度方向
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96)
    print("C2/C3. 沿墨量塌陷方向 (X_a = (1-a)X + a*白底) 的 loss / IoU / 梯度")
    print("=" * 96)
    hdr = (f"  {'α':>5}{'rawL2':>9}{'rawL1':>9}{'probeL2':>9}{'probeL1':>9}"
           f"{'骨架IoU':>9}{'墨量':>8}{'cos∇p2/∇r2':>12}{'cos∇p1/∇r1':>12}"
           f"{'cos∇p2/∇p1':>12}{'cos(-∇p2,真值)':>15}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for al in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        Xa = ((1 - al) * X + al * zbg).clone().requires_grad_(True)
        p2 = F.mse_loss(probe(Xa.to(DEV)).float().cpu(), Y)
        p1 = F.l1_loss(probe(Xa.to(DEV)).float().cpu(), Y)
        g2 = torch.autograd.grad(p2, Xa, retain_graph=True)[0].detach()
        g1 = torch.autograd.grad(p1, Xa, retain_graph=True)[0].detach()
        Xa = Xa.detach()
        with torch.no_grad():
            r2 = float(F.mse_loss(Xa, X)); r1 = float(F.l1_loss(Xa, X))
            gr2 = (Xa - X) / X.numel() * 2          # d rawL2 / d Xa
            gr1 = (Xa - X).sign() / X.numel()       # d rawL1 / d Xa
            iou_ = iou(skel_of(dec(Xa)), GT_SK)
            ink = float((dec(Xa) < 0.5).float().mean())
            _c = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            c_r2 = _c(g2, gr2)
            c_r1 = _c(g1, gr1)
            c_21 = _c(g2, g1)
            # 指向真值: -∇ 应与 (X - Xa) 同向
            c_true = _c(-g2, X - Xa)
        print(f"  {al:>5.2f}{r2:>9.5f}{r1:>9.5f}{float(p2):>9.5f}{float(p1):>9.5f}"
              f"{iou_:>9.4f}{ink:>8.4f}{c_r2:>12.4f}{c_r1:>12.4f}{c_21:>12.4f}{c_true:>15.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 动态范围: loss(坏)/loss(好) —— 越大越能提供梯度
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96)
    print("动态范围 (在**真实工作区间** α∈[0,0.3] 内, loss 涨了多少)")
    print("=" * 96)
    for nm, fn in [("rawL2", lambda Xa: float(F.mse_loss(Xa, X))),
                   ("rawL1", lambda Xa: float(F.l1_loss(Xa, X))),
                   ("probeL2", lambda Xa: float(F.mse_loss(probe(Xa.to(DEV)).float().cpu(), Y))),
                   ("probeL1", lambda Xa: float(F.l1_loss(probe(Xa.to(DEV)).float().cpu(), Y)))]:
        l0 = fn(X)
        l3 = fn(0.7 * X + 0.3 * zbg)
        print(f"  {nm:<9} loss(α=0)={l0:.5f}  loss(α=0.3)={l3:.5f}  "
              f"涨幅 {l3 - l0:+.5f}  相对 {l3 / max(l0, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
