# -*- coding: utf-8 -*-
"""
train_structnet_cpu.py — 训练 latent(4,32,32) -> 像素 skel/canny 的小网络 (纯 CPU)。

动机: latent 域训练时, 结构损失 (SkeletonLoss/EdgeGradientLoss) 需要像素级
skel/canny 图, 目前靠 vae.decode(整图) 或像素侧整图监督。本小网络学会
"只看 latent 就预测出对应的骨架/边缘图", 训练好后可:
  - 给 latent 训练提供轻量结构监督 (替代/辅助 VAE decode)
  - 或作为 ControlNet 的结构条件来源

数据 (5script/structnet/, mmap):
  latents.npy  f16 (N,4,32,32)
  skels.npy    u8  (N,256,256)  (0/255 -> 0/1)
  cannys.npy   u8  (N,256,256)
  ids.txt      每行 img_id

模型 (< 5M 参数):
  ConvResBlock x3 (4->64->64->64 @32) -> 像素上采样到 256 -> 2 输出头
  skel 头: sigmoid + BCE (稀疏, pos weight 处理不平衡)
  canny 头: sigmoid + BCE

用法:
  /opt/conda/bin/python tools/structnet/train_structnet_cpu.py \
      --data-dir 5script/structnet --epochs 30 --batch-size 64
"""
import os
import sys
import time
import glob
import json
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 小网络: latent -> skel/canny 像素图
# ---------------------------------------------------------------------------
class ConvResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = None
        if cin != cout or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return F.relu(h + x, inplace=True)


class LatentStructNet(nn.Module):
    """
    latent (4,32,32)  -> f (64,32,32) -> upsample -> skel_head (1,256,256) + canny_head (1,256,256)
    """
    def __init__(self, latent_ch=4, base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(latent_ch, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            ConvResBlock(base, base),
            ConvResBlock(base, base * 2),
            ConvResBlock(base * 2, base * 2),
            ConvResBlock(base * 2, base * 4, stride=2),   # 16x16
            ConvResBlock(base * 4, base * 4),
        )
        # 上采样到 256
        self.up1 = nn.Upsample(scale_factor=4, mode="nearest")
        self.up2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.head = nn.Conv2d(base * 4, 16, 3, 1, 1)
        self.skel_head = nn.Conv2d(16, 1, 1)
        self.canny_head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        f = self.stem(x)                      # (B,64,32,32)
        f = self.body(f)                      # (B,256,16,16)
        f = self.up1(f)                       # 64
        f = self.up2(f)                       # 256
        f = F.relu(self.head(f), inplace=True)  # (B,16,256,256)
        # raw logits (BCEWithLogits 内部做 sigmoid, pos_weight 支持稀疏正类)
        skel = self.skel_head(f)
        canny = self.canny_head(f)
        return skel, canny


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# 数据 (mmap)
# ---------------------------------------------------------------------------
class StructNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.lats = np.load(os.path.join(data_dir, "latents.npy"), mmap_mode="c")
        self.skels = np.load(os.path.join(data_dir, "skels.npy"), mmap_mode="c")
        self.cannys = np.load(os.path.join(data_dir, "cannys.npy"), mmap_mode="c")
        self.n = len(self.skels)
        log(f"[data] {self.n} samples; latents {self.lats.shape}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        lat = torch.from_numpy(np.asarray(self.lats[i], dtype=np.float32))      # (4,32,32)
        sk = torch.from_numpy((np.asarray(self.skels[i]) > 127).astype(np.float32)).unsqueeze(0)
        ca = torch.from_numpy((np.asarray(self.cannys[i]) > 127).astype(np.float32)).unsqueeze(0)
        return lat, sk, ca


# ---------------------------------------------------------------------------
# 指标: IoU + acc (skel)
# ---------------------------------------------------------------------------
def iou_binary(pred, gt, thr=0.5):
    p = (pred > thr).float()
    inter = (p * gt).sum()
    union = ((p + gt) > 0).sum()
    return (inter / (union + 1e-8)).item()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="5script/structnet")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-split", type=float, default=0.99, help="前 val_split 训练, 后 1-p 验证")
    ap.add_argument("--skel-pos-weight", type=float, default=20.0,
                    help="skel 密度 ~1%, 正类权重补偿")
    ap.add_argument("--out-dir", default="5script/results/structnet")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    ds = StructNetDataset(args.data_dir)
    n = len(ds)
    n_val = max(100, int(n * (1 - args.val_split)))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cpu")
    model = LatentStructNet().to(device)
    log(f"[model] params={count_params(model):,}  train={n_train} val={n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_pos = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.skel_pos_weight))
    bce_plain = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0)
        log(f"[resume] {args.resume} from epoch {start_epoch}")

    step = 0
    t0 = time.time()
    best_skel_iou = 0.0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for bi, (lat, sk, ca) in enumerate(train_loader):
            sk_logit, ca_logit = model(lat)
            loss_sk = bce_pos(sk_logit, sk)
            loss_ca = bce_plain(ca_logit, ca)
            # skel 稀疏更重要: 总 loss = loss_ca + 2*loss_sk
            loss = loss_ca + 2.0 * loss_sk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                sps = args.log_every / max(dt, 1e-6)
                t0 = time.time()
                log(f"ep{epoch} step{step}: loss={running/args.log_every:.4f} "
                    f"(sk {loss_sk.item():.4f} ca {loss_ca.item():.4f}) {sps:.1f} it/s")
                running = 0.0

        # ---- 验证 ----
        model.eval()
        sk_iou = ca_iou = 0.0
        sk_acc = ca_acc = 0.0
        cnt = 0
        with torch.no_grad():
            for lat, sk, ca in val_loader:
                sk_logit, ca_logit = model(lat)
                sk_pred = torch.sigmoid(sk_logit)
                ca_pred = torch.sigmoid(ca_logit)
                sk_iou += iou_binary(sk_pred, sk)
                ca_iou += iou_binary(ca_pred, ca)
                sk_acc += ((sk_pred > 0.5).float() == sk).float().mean().item()
                ca_acc += ((ca_pred > 0.5).float() == ca).float().mean().item()
                cnt += 1
        sk_iou /= cnt; ca_iou /= cnt; sk_acc /= cnt; ca_acc /= cnt
        log(f"== epoch {epoch}: skel IoU={sk_iou:.4f} acc={sk_acc:.4f} | "
            f"canny IoU={ca_iou:.4f} acc={ca_acc:.4f}")

        if sk_iou > best_skel_iou:
            best_skel_iou = sk_iou
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                        "epoch": epoch, "skel_iou": sk_iou, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))
            log(f"[save] best.pt skel_iou={sk_iou:.4f}")
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                    "epoch": epoch, "skel_iou": sk_iou},
                   os.path.join(args.out_dir, "last.pt"))

    log("Done!")


if __name__ == "__main__":
    main()