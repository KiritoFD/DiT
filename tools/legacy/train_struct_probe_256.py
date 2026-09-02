# -*- coding: utf-8 -*-
"""
train_struct_probe_256.py — 训练 latent(4,32,32) → skel/canny(1,256,256) 小网络 (CPU)。

直接复用 MCCDLatentDataset (已处理好 latent shards + canny/skel PNG 加载),
不需要额外 cache 打包。skel/canny GT 已存在于 final_canny/ final_skeleton/ (从图算好的)。

网络 (<5M 参数):
  stem Conv(4→64) + 3×_ResidualConv(64) @32
  → 3 阶 PixelShuffle 上采样到 256
  → 双头 (canny_logit, skel_logit)

损失: BCEWithLogits(pos_weight 处理稀疏) + soft Dice
评估: val IoU / acc (阈值 0.5)

用法:
  /opt/conda/bin/python tools/train_struct_probe_256.py \
      --csv 5script/train.csv --epochs 8 --batch-size 64
"""
import os
import sys
import time
import json
import argparse
import datetime

# 让 src 模块可 import
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 网络: latent(4,32,32) → (2, 256, 256) logits (canny, skel)
# ---------------------------------------------------------------------------
class _ResidualConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class StructProbe256(nn.Module):
    """latent(4,32,32) → (canny_logit, skel_logit) 各 (1,256,256)。
    三阶 PixelShuffle 上采样: 32→64→128→256。"""
    def __init__(self, in_ch=4, base=64, depth=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        self.body = nn.Sequential(*[_ResidualConv(base) for _ in range(depth)])
        # 32→64
        self.up1 = nn.Sequential(
            nn.Conv2d(base, base * 4, 3, padding=1), nn.SiLU(),
            nn.PixelShuffle(2))  # → base @ 64
        # 64→128
        self.up2 = nn.Sequential(
            nn.Conv2d(base, base * 4, 3, padding=1), nn.SiLU(),
            nn.PixelShuffle(2))  # → base @ 128
        # 128→256
        self.up3 = nn.Sequential(
            nn.Conv2d(base, base * 4, 3, padding=1), nn.SiLU(),
            nn.PixelShuffle(2))  # → base @ 256
        # 双头共享 trunk
        self.head = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                   nn.Conv2d(base, 16, 3, padding=1), nn.SiLU())
        self.canny_head = nn.Conv2d(16, 1, 1)
        self.skel_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        x = self.head(x)
        return self.canny_head(x), self.skel_head(x)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def bce_dice(logits, target, pos_weight):
    """logits/target: (B,1,H,W); pos_weight scalar 张量."""
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
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--canny-root", default="final_canny")
    ap.add_argument("--skel-root", default="final_skeleton")
    ap.add_argument("--out-dir", default="5script/results/structnet256")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--skel-pos-weight", type=float, default=80.0)
    ap.add_argument("--canny-pos-weight", type=float, default=10.0)
    ap.add_argument("--num-threads", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)

    # ---- 数据: 复用 MCCDLatentDataset (已处理 latent shards + 256 canny/skel) ----
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

    # 固定 val 子集 (前 val_n 行, 与训练分开)
    val_n = min(args.val_n, n)
    train_idx = list(range(val_n, n))
    val_idx = list(range(val_n))
    log(f"[split] train={len(train_idx)} val={val_n}")

    # 预取 val 张量 (CPU 一次加载)
    log("[val] preloading val tensors ...")
    lat_v = torch.zeros(val_n, 4, 32, 32)
    can_v = torch.zeros(val_n, 1, 256, 256)
    sk_v = torch.zeros(val_n, 1, 256, 256)
    for i, gi in enumerate(val_idx):
        item = ds[gi]
        lat_v[i] = item['latent']
        can_v[i] = item['canny']
        sk_v[i] = item['skeleton']
    log(f"[val] done: lat {lat_v.shape} can {can_v.shape}")

    # ---- 模型 ----
    device = torch.device("cpu")
    model = StructProbe256(in_ch=4, base=args.base, depth=args.depth).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"[model] StructProbe256 base={args.base} depth={args.depth} params={nparams:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    pw_can = torch.tensor(args.canny_pos_weight)
    pw_sk = torch.tensor(args.skel_pos_weight)

    best_sk_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        # 每 epoch 重新 shuffle 训练索引
        perm = np.random.permutation(train_idx)
        running = 0.0; n_batch = 0; t0 = time.time()
        for bi in range(0, len(perm), args.batch_size):
            idx_b = perm[bi:bi + args.batch_size]
            # 组装 batch (单进程, 直接索引 ds)
            lat = torch.stack([ds[int(i)]['latent'] for i in idx_b])
            can = torch.stack([ds[int(i)]['canny'] for i in idx_b])
            sk = torch.stack([ds[int(i)]['skeleton'] for i in idx_b])

            can_logit, sk_logit = model(lat)
            loss = bce_dice(can_logit, can, pw_can) + bce_dice(sk_logit, sk, pw_sk)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item(); n_batch += 1
            g = bi // args.batch_size + 1
            if g % args.log_every == 0:
                dt = time.time() - t0
                log(f"ep{epoch} b{g}: loss={running/n_batch:.4f} {g*args.batch_size/ (n-val_n) * 100:.1f}% "
                    f"{g*args.batch_size/max(dt,1e-6):.0f} sample/s")
                running = 0.0; n_batch = 0; t0 = time.time()

        # ---- val ----
        model.eval()
        sk_iou = can_iou = sk_acc = can_acc = 0.0; cnt = 0
        with torch.no_grad():
            for bi in range(0, val_n, args.batch_size):
                lat = lat_v[bi:bi + args.batch_size]
                can = can_v[bi:bi + args.batch_size]
                sk = sk_v[bi:bi + args.batch_size]
                can_logit, sk_logit = model(lat)
                can_p = can_logit.sigmoid(); sk_p = sk_logit.sigmoid()
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
                                "canny_iou": can_iou,
                                "skel_acc": sk_acc, "canny_acc": can_acc}) + "\n")
    log("Done!")


if __name__ == "__main__":
    main()