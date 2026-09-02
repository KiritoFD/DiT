# -*- coding: utf-8 -*-
"""
train_skel_decoder.py — 简洁: latent(4,32,32) → skel(1,256,256) 单网络。

设计极简: 几层 conv @32 提特征 → bilinear 上采样到 256 → 1×1 conv 出 skel logit。
所有重计算在 32×32, 256 只有一个 1×1 conv。可导, 可端到端微调。
"""
import os, sys, time, json, argparse, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")


def log(m):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


class SkelDecoder(nn.Module):
    """latent(4,32,32) → skel_logit(1,256,256). ~200K params."""
    def __init__(self, base=64, depth=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, base, 3, padding=1), nn.SiLU())
        blocks = []
        for _ in range(depth):
            blocks += [
                nn.GroupNorm(8, base), nn.SiLU(),
                nn.Conv2d(base, base, 3, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(base, 1, 1)  # 1×1 conv @32 → logit

    def forward(self, x):
        f = self.stem(x)
        f = self.body(f)            # (B,base,32,32)
        logit = self.head(f)        # (B,1,32,32)
        # bilinear 上采样到 256 (零参数, 可导)
        return F.interpolate(logit, size=256, mode="bilinear", align_corners=False)


def bce_dice(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = logits.sigmoid()
    eps = 1e-6
    dice = 1.0 - (2.0 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)
    return bce + dice


def iou(pred, gt, thr=0.5, eps=1e-6):
    p = (pred > thr).float()
    i = (p * gt).sum()
    u = ((p + gt) > 0).float().sum()
    return (i / (u + eps)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--skel-root", default="final_skeleton",
                    help="1px GT=final_skeleton; 3px 膨胀版=final_skeleton_d3")
    ap.add_argument("--out-dir", default="5script/results/skel_decoder")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--pos-weight", type=float, default=80.0)
    ap.add_argument("--num-threads", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(0)

    from latent_dataset import MCCDLatentDataset
    log("[data] loading latent + skel ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=None, skel_root=args.skel_root,
        image_size=256, load_skel=True,
        is_train=True, preload=True, load_image=False, load_canny=False,
        structure_size=256, num_preload_workers=16)
    n = len(ds)
    log(f"[data] {n} samples")

    val_n = min(args.val_n, n)
    vi = np.arange(val_n)
    ti = np.arange(val_n, n)
    log(f"[split] train={len(ti)} val={val_n}")

    # val 预取
    lat_v = torch.from_numpy(ds._latents[vi].copy())
    sk_v = torch.from_numpy((ds._skels[vi] > 127).astype(np.float32)).unsqueeze(1)
    log(f"[val] {lat_v.shape} {sk_v.shape}")

    model = SkelDecoder(base=args.base, depth=args.depth)
    log(f"[model] params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw = torch.tensor(args.pos_weight)

    best_iou = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(ti)
        run = 0.0; nb = 0; t0 = time.time()
        for bi in range(0, len(perm), args.batch_size):
            idx = perm[bi:bi + args.batch_size]
            lat = torch.from_numpy(ds._latents[idx].copy())
            sk = torch.from_numpy((ds._skels[idx] > 127).astype(np.float32)).unsqueeze(1)
            logit = model(lat)
            loss = bce_dice(logit, sk, pw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1
            if nb % args.log_every == 0:
                log(f"ep{ep} b{nb}: loss={run/nb:.4f} {nb*args.batch_size/max(time.time()-t0,1e-6):.0f} s/s")
                run = 0.0; t0 = time.time()

        # val
        model.eval()
        si = 0.0; cnt = 0
        with torch.no_grad():
            for bi in range(0, val_n, args.batch_size):
                logit = model(lat_v[bi:bi + args.batch_size])
                si += iou(logit.sigmoid(), sk_v[bi:bi + args.batch_size]); cnt += 1
        si /= cnt
        log(f"== ep{ep}: skel IoU={si:.4f}")
        ck = {"model": model.state_dict(), "epoch": ep, "skel_iou": si}
        torch.save(ck, os.path.join(args.out_dir, "last.pt"))
        if si > best_iou:
            best_iou = si
            torch.save(ck, os.path.join(args.out_dir, "best.pt"))
            log(f"[save] best IoU={si:.4f}")
        with open(os.path.join(args.out_dir, "history.json"), "a") as f:
            f.write(json.dumps({"epoch": ep, "skel_iou": si}) + "\n")
    log("Done!")


if __name__ == "__main__":
    main()