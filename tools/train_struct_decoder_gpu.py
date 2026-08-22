# -*- coding: utf-8 -*-
"""
train_struct_decoder_gpu.py — GPU 快速训练 latent→skel 和 latent→canny 两个独立网络。

v2 升级:
  - 多尺度 pixel-shuffle 解码器 (32→64→128→256), 替代 bilinear, 预测更锐利
  - PixelShuffle = reshape+permute, 零损失梯度透传, 全程可导
  - Cosine LR schedule
  - 验证指标: IoU + Precision + Recall + 密度校准
  - 训练后自动跑梯度健康检查 (梯度方向/幅度/噪声鲁棒性)

梯度路径: loss@256 → head(1×1) → up3(PixelShuffle+conv@128) → up2(@64) → up1(@32) → body → stem → latent
"""
import os, sys, time, json, argparse, datetime, math, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")


def log(m):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


class StructDecoder(nn.Module):
    """latent(4,32,32) → logit(1,256,256).

    多尺度 pixel-shuffle: 32→64→128→256, 每级 1×1 conv + PixelShuffle(2) + SiLU.
    PixelShuffle 是纯 reshape+permute, 梯度完美透传.
    全程可导, ~400K params (base=96, depth=6).
    """
    def __init__(self, in_ch=4, base=64, depth=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        blocks = []
        for _ in range(depth):
            blocks += [
                nn.GroupNorm(8, base), nn.SiLU(),
                nn.Conv2d(base, base, 3, padding=1)]
        self.body = nn.Sequential(*blocks)
        # 32→64→128→256, 每级: 1×1 conv (base→base*4) + PixelShuffle(2) + SiLU
        # up1/up2 加 3×3 conv 提升特征, up3 只 PixelShuffle+SiLU (256×256 不做 3×3 省显存)
        self.up1 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Conv2d(base, base * 4, 1), nn.PixelShuffle(2), nn.SiLU())
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f = self.stem(x)       # (B,base,32,32)
        f = self.body(f)       # (B,base,32,32)
        f = self.up1(f)        # (B,base,64,64)
        f = self.up2(f)        # (B,base,128,128)
        f = self.up3(f)        # (B,base,256,256)
        return self.head(f)    # (B,1,256,256)


def bce_dice(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = logits.sigmoid()
    eps = 1e-6
    dice = 1.0 - (2.0 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)
    return bce + dice


def metrics(pred, gt, thr=0.5, eps=1e-6):
    """返回 IoU, Precision, Recall, pred_density, gt_density."""
    p = (pred > thr).float()
    tp = (p * gt).sum()
    fp = (p * (1 - gt)).sum()
    fn = ((1 - p) * gt).sum()
    iou = (tp / (tp + fp + fn + eps)).item()
    prec = (tp / (tp + fp + eps)).item()
    rec = (tp / (tp + fn + eps)).item()
    pd = p.mean().item()
    gd = gt.mean().item()
    return iou, prec, rec, pd, gd


class BatchDataLoader:
    def __init__(self, latents, maps, idx, batch_size, device):
        self.latents = latents
        self.maps = maps
        self.idx = np.array(idx)
        self.bs = batch_size
        self.device = device

    def __iter__(self):
        perm = np.random.permutation(self.idx)
        for bi in range(0, len(perm), self.bs):
            idx_b = perm[bi:bi + self.bs]
            lat = torch.from_numpy(self.latents[idx_b].copy()).to(self.device)
            mp = torch.from_numpy((self.maps[idx_b] > 127).astype(np.float32)).unsqueeze(1).to(self.device)
            yield lat, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="5script/train_full.csv")
    ap.add_argument("--latent-shards-dir", default="final_latents")
    ap.add_argument("--skel-root", default="final_skeleton_d3")
    ap.add_argument("--canny-root", default="final_canny_d3")
    ap.add_argument("--out-dir", default="5script/results/struct_decoder_gpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=2000)
    ap.add_argument("--skel-pos-weight", type=float, default=15.0)
    ap.add_argument("--canny-pos-weight", type=float, default=8.0)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # ---- 数据 ----
    from latent_dataset import MCCDLatentDataset
    log("[data] loading latent + skel(3px) + canny(3px) ...")
    ds = MCCDLatentDataset(
        csv_file=args.csv, latent_shards_dir=args.latent_shards_dir,
        img_root=None, canny_root=args.canny_root, skel_root=args.skel_root,
        image_size=256, load_canny=True, load_skel=True,
        is_train=True, preload=True, load_image=False,
        structure_size=256, num_preload_workers=16)
    n = len(ds)
    log(f"[data] {n} samples (full 329k)")

    val_n = min(args.val_n, n)
    vi = np.arange(val_n)
    ti = np.arange(val_n, n)
    log(f"[split] train={len(ti)} val={val_n}")

    lat_v = torch.from_numpy(ds._latents[vi].copy()).to(device)
    sk_v = torch.from_numpy((ds._skels[vi] > 127).astype(np.float32)).unsqueeze(1).to(device)
    ca_v = torch.from_numpy((ds._cannys[vi] > 127).astype(np.float32)).unsqueeze(1).to(device)
    log(f"[val] {lat_v.shape}")

    # ---- 网络 ----
    skel_net = StructDecoder(in_ch=4, base=args.base, depth=args.depth).to(device)
    canny_net = StructDecoder(in_ch=4, base=args.base, depth=args.depth).to(device)
    p_sk = sum(p.numel() for p in skel_net.parameters())
    p_ca = sum(p.numel() for p in canny_net.parameters())
    log(f"[model] skel params={p_sk:,} canny params={p_ca:,}")

    opt_sk = torch.optim.AdamW(skel_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_ca = torch.optim.AdamW(canny_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine LR with warmup
    def get_lr(epoch):
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    pw_sk = torch.tensor(args.skel_pos_weight, device=device)
    pw_ca = torch.tensor(args.canny_pos_weight, device=device)

    skel_loader = BatchDataLoader(ds._latents, ds._skels, ti, args.batch_size, device)
    canny_loader = BatchDataLoader(ds._latents, ds._cannys, ti, args.batch_size, device)

    best_sk_iou = 0.0
    best_ca_iou = 0.0
    for ep in range(args.epochs):
        lr = get_lr(ep)
        for g in opt_sk.param_groups: g['lr'] = lr
        for g in opt_ca.param_groups: g['lr'] = lr

        # ---- skel ----
        skel_net.train()
        run = 0.0; nb = 0; t0 = time.time()
        for lat, sk in skel_loader:
            logit = skel_net(lat)
            loss = bce_dice(logit, sk, pw_sk)
            opt_sk.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(skel_net.parameters(), 1.0)
            opt_sk.step()
            run += loss.item(); nb += 1
            if nb % args.log_every == 0:
                log(f"[skel] ep{ep} b{nb}: loss={run/nb:.4f} lr={lr:.2e} {nb*args.batch_size/max(time.time()-t0,1e-6):.0f} s/s")
                run = 0.0; t0 = time.time()

        # ---- canny ----
        canny_net.train()
        run = 0.0; nb = 0; t0 = time.time()
        for lat, ca in canny_loader:
            logit = canny_net(lat)
            loss = bce_dice(logit, ca, pw_ca)
            opt_ca.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(canny_net.parameters(), 1.0)
            opt_ca.step()
            run += loss.item(); nb += 1
            if nb % args.log_every == 0:
                log(f"[canny] ep{ep} b{nb}: loss={run/nb:.4f} lr={lr:.2e} {nb*args.batch_size/max(time.time()-t0,1e-6):.0f} s/s")
                run = 0.0; t0 = time.time()

        # ---- val ----
        skel_net.eval(); canny_net.eval()
        sk_res = [0.0]*5; ca_res = [0.0]*5; cnt = 0
        with torch.no_grad():
            for bi in range(0, val_n, args.batch_size):
                lat = lat_v[bi:bi + args.batch_size]
                sk = sk_v[bi:bi + args.batch_size]
                ca = ca_v[bi:bi + args.batch_size]
                sk_p = skel_net(lat).sigmoid()
                ca_p = canny_net(lat).sigmoid()
                si, sp, sr, spd, sgd = metrics(sk_p, sk)
                ci, cp, cr, cpd, cgd = metrics(ca_p, ca)
                for j in range(5): sk_res[j] += [si,sp,sr,spd,sgd][j]
                for j in range(5): ca_res[j] += [ci,cp,cr,cpd,cgd][j]
                cnt += 1
        sk_res = [v/cnt for v in sk_res]
        ca_res = [v/cnt for v in ca_res]
        log(f"== ep{ep} (lr={lr:.2e}): "
            f"skel IoU={sk_res[0]:.4f} P={sk_res[1]:.4f} R={sk_res[2]:.4f} dens={sk_res[3]:.4f}/{sk_res[4]:.4f} | "
            f"canny IoU={ca_res[0]:.4f} P={ca_res[1]:.4f} R={ca_res[2]:.4f} dens={ca_res[3]:.4f}/{ca_res[4]:.4f}")

        ck_sk = {"model": skel_net.state_dict(), "epoch": ep, "iou": sk_res[0],
                 "precision": sk_res[1], "recall": sk_res[2], "config": vars(args), "type": "skel"}
        ck_ca = {"model": canny_net.state_dict(), "epoch": ep, "iou": ca_res[0],
                 "precision": ca_res[1], "recall": ca_res[2], "config": vars(args), "type": "canny"}
        torch.save(ck_sk, os.path.join(args.out_dir, "skel_last.pt"))
        torch.save(ck_ca, os.path.join(args.out_dir, "canny_last.pt"))
        if sk_res[0] > best_sk_iou:
            best_sk_iou = sk_res[0]
            torch.save(ck_sk, os.path.join(args.out_dir, "skel_best.pt"))
            log(f"[save] skel_best IoU={sk_res[0]:.4f} R={sk_res[2]:.4f}")
        if ca_res[0] > best_ca_iou:
            best_ca_iou = ca_res[0]
            torch.save(ck_ca, os.path.join(args.out_dir, "canny_best.pt"))
            log(f"[save] canny_best IoU={ca_res[0]:.4f} R={ca_res[2]:.4f}")
        with open(os.path.join(args.out_dir, "history.json"), "a") as f:
            f.write(json.dumps({"epoch": ep, "lr": lr,
                "skel_iou": sk_res[0], "skel_prec": sk_res[1], "skel_rec": sk_res[2],
                "skel_pred_dens": sk_res[3], "skel_gt_dens": sk_res[4],
                "canny_iou": ca_res[0], "canny_prec": ca_res[1], "canny_rec": ca_res[2],
                "canny_pred_dens": ca_res[3], "canny_gt_dens": ca_res[4]}) + "\n")

    log("=== Training done. Running gradient health check... ===")
    _gradient_health_check(skel_net, canny_net, lat_v, sk_v, ca_v, device, args.out_dir)
    log("All done!")


def _gradient_health_check(skel_net, canny_net, lat_v, sk_v, ca_v, device, out_dir):
    """验收: 梯度方向/幅度/噪声鲁棒性."""
    skel_net.eval(); canny_net.eval()
    n_check = min(200, lat_v.shape[0])
    lat = lat_v[:n_check].clone().requires_grad_(True)
    sk = sk_v[:n_check]
    ca = ca_v[:n_check]

    # 1. 梯度方向: GT latent → loss → ∂L/∂latent, 检查空间分布
    log("[grad-check] 1. 梯度方向测试 (GT latent)")
    sk_logit = skel_net(lat)
    sk_loss = F.binary_cross_entropy_with_logits(sk_logit, sk)
    g_sk = torch.autograd.grad(sk_loss, lat, retain_graph=False)[0]
    # 梯度空间集中度: 笔画区梯度 vs 非笔画区梯度
    # latent 空间没有直接的笔画 mask, 用梯度 norm 的空间分布衡量
    g_sk_per_sample = g_sk.view(n_check, -1).norm(dim=1)  # (N,)
    g_sk_mean = g_sk_per_sample.mean().item()
    g_sk_std = g_sk_per_sample.std().item()
    log(f"  skel: grad_norm mean={g_sk_mean:.4f} std={g_sk_std:.4f} "
        f"min={g_sk_per_sample.min():.4f} max={g_sk_per_sample.max():.4f}")

    lat2 = lat_v[:n_check].clone().requires_grad_(True)
    ca_logit = canny_net(lat2)
    ca_loss = F.binary_cross_entropy_with_logits(ca_logit, ca)
    g_ca = torch.autograd.grad(ca_loss, lat2)[0]
    g_ca_per_sample = g_ca.view(n_check, -1).norm(dim=1)
    log(f"  canny: grad_norm mean={g_ca_per_sample.mean().item():.4f} "
        f"std={g_ca_per_sample.std().item():.4f}")

    # 2. 噪声鲁棒性: GT latent + σ噪声 → IoU 衰减
    log("[grad-check] 2. 噪声鲁棒性 (IoU vs latent noise σ)")
    noise_results = {"skel": {}, "canny": {}}
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        lat_n = lat_v[:n_check] + torch.randn_like(lat_v[:n_check]) * sigma
        with torch.no_grad():
            sk_p = skel_net(lat_n).sigmoid()
            ca_p = canny_net(lat_n).sigmoid()
            si, _, _, _, _ = metrics(sk_p, sk)
            ci, _, _, _, _ = metrics(ca_p, ca)
        noise_results["skel"][sigma] = round(si, 4)
        noise_results["canny"][sigma] = round(ci, 4)
        log(f"  σ={sigma:.2f}: skel IoU={si:.4f} | canny IoU={ci:.4f}")

    # 3. 梯度幅度 vs 主任务 (diff loss 梯度量级)
    log("[grad-check] 3. 梯度幅度 (struct loss vs random latent L2)")
    lat3 = lat_v[:n_check].clone().requires_grad_(True)
    sk_logit3 = skel_net(lat3)
    sk_loss3 = F.binary_cross_entropy_with_logits(sk_logit3, sk)
    g3 = torch.autograd.grad(sk_loss3, lat3)[0]
    # 模拟 diff loss: ||latent||^2 的梯度
    g_diff = 2 * lat3.detach()
    ratio = g3.view(n_check, -1).norm(dim=1).mean() / (g_diff.view(n_check, -1).norm(dim=1).mean() + 1e-8)
    log(f"  struct_grad_norm / latent_norm = {ratio.item():.4f} "
        f"(struct_grad={g3.view(n_check,-1).norm(dim=1).mean().item():.4f}, "
        f"latent_norm={g_diff.view(n_check,-1).norm(dim=1).mean().item():.4f})")

    report = {
        "skel_grad_norm": {"mean": round(g_sk_mean, 4), "std": round(g_sk_std, 4)},
        "canny_grad_norm": {"mean": round(g_ca_per_sample.mean().item(), 4),
                            "std": round(g_ca_per_sample.std().item(), 4)},
        "noise_robustness": noise_results,
        "grad_to_latent_ratio": round(ratio.item(), 4),
    }
    with open(os.path.join(out_dir, "gradient_health.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"[grad-check] Report saved to {out_dir}/gradient_health.json")

    # 验收判定
    sk_clean = noise_results["skel"][0.0]
    sk_noisy = noise_results["skel"][0.2]
    ca_clean = noise_results["canny"][0.0]
    ca_noisy = noise_results["canny"][0.2]
    verdict = []
    verdict.append(f"IoU(skel,clean)={sk_clean:.4f} {'✅' if sk_clean >= 0.75 else '❌'} (≥0.75)")
    verdict.append(f"IoU(canny,clean)={ca_clean:.4f} {'✅' if ca_clean >= 0.70 else '❌'} (≥0.70)")
    verdict.append(f"噪声鲁棒 σ=0.2: skel={sk_noisy:.4f} ({'✅' if sk_noisy >= 0.5 else '❌'} ≥0.5) "
                   f"canny={ca_noisy:.4f} ({'✅' if ca_noisy >= 0.4 else '❌'} ≥0.4)")
    verdict.append(f"梯度幅度比={ratio.item():.4f} ({'✅' if 0.01 <= ratio.item() <= 100 else '⚠️'} ∈[0.01,100])")
    log("=== 验收结果 ===")
    for v in verdict:
        log(f"  {v}")


if __name__ == "__main__":
    main()