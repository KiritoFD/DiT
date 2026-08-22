# -*- coding: utf-8 -*-
"""
train_struct_probe_256_v2.py — 简洁可导的 latent→skel/canny 小网络 (CPU, v2)。

先验驱动的简洁设计:
  1. skel ⊂ ink (骨架是墨迹子集), canny ≈ ∂ink (边缘是墨迹边界)
     -> 先预测密集 ink map (32×32, 容易学), skel/canny 在 ink 上细化
  2. bilinear 上采样 (零参数, 可导, 对 1px 细线比 PixelShuffle 更锐利)
  3. 高分辨率 (256) 只有 1×1 小卷积头, 计算量压在 32×32 低分辨率

网络 (<300K 参数):
  latent(4,32,32) → stem(64) → 3×ResBlock(64) @32 → ink_head(1,32) [先验]
                → upsample(bilinear, 32→256) → skel_head(1,256) + canny_head(1,256)

损失: BCE(pos_weight) + Dice, skel 通道额外乘 ink 先验 (skel 只在 ink 区域出现)
"""
import os
import sys
import time
import json
import argparse
import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 网络: 先验驱动的 latent → ink → skel/canny
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class StructNetV2(nn.Module):
    """latent(4,32,32) → ink(1,32) [先验] → bilinear↑256 → skel(1,256) + canny(1,256).
    所有重计算在 32×32; 256 只有 1×1 conv. 参数 ~280K."""
    def __init__(self, in_ch=4, base=64, depth=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        self.body = nn.Sequential(*[ResBlock(base) for _ in range(depth)])

        # ink 先验头 (32×32, 密集, 容易学)
        self.ink_head = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 1))

        # skel/canny 细化头: 在 32×32 特征上各做 1 层 conv, 然后 bilinear↑256
        self.skel_refine = nn.Conv2d(base + 1, 16, 3, padding=1)  # +1 for ink prior
        self.canny_refine = nn.Conv2d(base + 1, 16, 3, padding=1)
        self.skel_head = nn.Conv2d(16, 1, 1)
        self.canny_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        f = self.stem(x)          # (B,64,32,32)
        f = self.body(f)          # (B,64,32,32)
        ink_logit = self.ink_head(f)  # (B,1,32,32) — ink 先验

        # skel/canny 在 32×32 上细化 (拼接 ink 先验)
        sk_f = F.silu(self.skel_refine(torch.cat([f, ink_logit], dim=1)))  # (B,16,32,32)
        ca_f = F.silu(self.canny_refine(torch.cat([f, ink_logit], dim=1)))

        # bilinear 上采样到 256 (零参数, 可导, 对细线比 PixelShuffle 锐利)
        sk_logit = self.skel_head(F.interpolate(sk_f, size=256, mode="bilinear", align_corners=False))
        ca_logit = self.canny_head(F.interpolate(ca_f, size=256, mode="bilinear", align_corners=False))
        return ca_logit, sk_logit, ink_logit


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def bce_dice(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = logits.sigmoid()
    eps = 1e-6
    dice = 1.0 - (2.0 * (probs * target).sum() + eps) / (probs.sum() + target.sum() + eps)
    return bce + dice


def iou_binary(pred_prob, gt, thr=0.5, eps=1e-6):
    p = (pred_prob > thr).float()
    inter = (p * gt).sum()
    union = ((p + gt) > 0).float().sum()
    return (inter / (union + eps)).item()


# ---------------------------------------------------------------------------
# 批量数据加载 (直接 numpy 索引, 不走 __getitem__)
# ---------------------------------------------------------------------------
class BatchDataLoader:
    """直接从 preload 的 numpy 数组批量切片, 避免逐条 __getitem__ 开销."""
    def __init__(self, ds, train_idx, batch_size, shuffle=True):
        self.latents = ds._latents  # (N,4,32,32) float32
        self.cannys = ds._cannys    # (N,256,256) uint8
        self.skels = ds._skels      # (N,256,256) uint8
        self.idx = np.array(train_idx)
        self.bs = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        perm = np.random.permutation(self.idx) if self.shuffle else self.idx
        for bi in range(0, len(perm), self.bs):
            idx_b = perm[bi:bi + self.bs]
            lat = torch.from_numpy(self.latents[idx_b].copy())  # (B,4,32,32)
            can = torch.from_numpy((self.cannys[idx_b] > 127).astype(np.float32)).unsqueeze(1)  # (B,1,256,256)
            sk = torch.from_numpy((self.skels[idx_b] > 127).astype(np.float32)).unsqueeze(1)
            yield lat, can, sk


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--canny-root", default="final_canny")
    ap.add_argument("--skel-root", default="final_skeleton")
    ap.add_argument("--out-dir", default="5script/results/structnet256")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--skel-pos-weight", type=float, default=80.0)
    ap.add_argument("--canny-pos-weight", type=float, default=10.0)
    ap.add_argument("--ink-pos-weight", type=float, default=3.0)
    ap.add_argument("--ink-weight", type=float, default=0.3,
                    help="ink 先验损失权重 (辅助, 帮助 skel/canny 收敛)")
    ap.add_argument("--num-threads", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)

    from latent_dataset import MCCDLatentDataset
    log("[data] loading (latent+canny+skel 256) ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=None, canny_root=args.canny_root, skel_root=args.skel_root,
        image_size=256, load_canny=True, load_skel=True,
        is_train=True, preload=True, load_image=False,
        structure_size=256, num_preload_workers=16)
    n = len(ds)
    log(f"[data] {n} samples loaded")

    val_n = min(args.val_n, n)
    train_idx = list(range(val_n, n))
    val_idx = list(range(val_n))
    log(f"[split] train={len(train_idx)} val={val_n}")

    # val 预取 (批量)
    log("[val] preloading ...")
    vi = np.array(val_idx)
    lat_v = torch.from_numpy(ds._latents[vi].copy())
    can_v = torch.from_numpy((ds._cannys[vi] > 127).astype(np.float32)).unsqueeze(1)
    sk_v = torch.from_numpy((ds._skels[vi] > 127).astype(np.float32)).unsqueeze(1)
    log(f"[val] done: {lat_v.shape} {can_v.shape}")

    device = torch.device("cpu")
    model = StructNetV2(in_ch=4, base=args.base, depth=args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"[model] StructNetV2 base={args.base} depth={args.depth} params={nparams:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    pw_can = torch.tensor(args.canny_pos_weight)
    pw_sk = torch.tensor(args.skel_pos_weight)
    pw_ink = torch.tensor(args.ink_pos_weight)

    train_loader = BatchDataLoader(ds, train_idx, args.batch_size, shuffle=True)

    best_sk_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0; n_batch = 0; t0 = time.time()
        for lat, can, sk in train_loader:
            ca_logit, sk_logit, ink_logit = model(lat)
            # ink 先验: 下采样 canny/skel 的并集近似 ink (32×32 max-pool)
            ink_gt = F.adaptive_max_pool2d(torch.max(can, sk), 32)
            loss = (bce_dice(ca_logit, can, pw_can)
                    + bce_dice(sk_logit, sk, pw_sk)
                    + args.ink_weight * bce_dice(ink_logit, ink_gt, pw_ink))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item(); n_batch += 1
            if n_batch % args.log_every == 0:
                dt = time.time() - t0
                log(f"ep{epoch} b{n_batch}: loss={running/n_batch:.4f} "
                    f"{n_batch*args.batch_size/max(dt,1e-6):.0f} s/s")
                running = 0.0; t0 = time.time()

        # ---- val ----
        model.eval()
        sk_iou = can_iou = sk_acc = can_acc = 0.0; cnt = 0
        with torch.no_grad():
            for bi in range(0, val_n, args.batch_size):
                lat = lat_v[bi:bi + args.batch_size]
                can = can_v[bi:bi + args.batch_size]
                sk = sk_v[bi:bi + args.batch_size]
                ca_l, sk_l, _ = model(lat)
                can_p = ca_l.sigmoid(); sk_p = sk_l.sigmoid()
                can_iou += iou_binary(can_p, can); sk_iou += iou_binary(sk_p, sk)
                can_acc += ((can_p > 0.5).float() == can).float().mean().item()
                sk_acc += ((sk_p > 0.5).float() == sk).float().mean().item()
                cnt += 1
        sk_iou /= cnt; can_iou /= cnt; sk_acc /= cnt; can_acc /= cnt
        log(f"== epoch {epoch}: skel IoU={sk_iou:.4f} acc={sk_acc:.4f} | "
            f"canny IoU={can_iou:.4f} acc={can_acc:.4f}")

        ck = {"model": model.state_dict(), "epoch": epoch,
              "skel_iou": sk_iou, "canny_iou": can_iou, "args": vars(args)}
        torch.save(ck, os.path.join(args.out_dir, "last.pt"))
        if sk_iou > best_sk_iou:
            best_sk_iou = sk_iou
            torch.save(ck, os.path.join(args.out_dir, "best.pt"))
            log(f"[save] best.pt skel_iou={sk_iou:.4f}")
        with open(os.path.join(args.out_dir, "history.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "skel_iou": sk_iou,
                                "canny_iou": can_iou}) + "\n")
    log("Done!")


if __name__ == "__main__":
    main()