# -*- coding: utf-8 -*-
"""
手动梯度健康检查 + 显存开销评估脚本.

加载 skel_best.pt / canny_best.pt, 运行:
  1. 梯度方向测试 (GT latent → ∂L/∂latent, 空间分布)
  2. 噪声鲁棒性 (σ=0~0.5, IoU 衰减)
  3. 梯度幅度 vs latent norm
  4. 显存开销: decoder forward+backward 的峰值显存 vs VAE decode
  5. 小模型对比: base=32 depth=3 的同测试
"""
import os, sys, json, time
os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, "/root/Workspace/xy/DiT/src")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from latent_dataset import MCCDLatentDataset

sys.stdout.reconfigure(encoding="utf-8")

DEVICE = "cuda"
OUT_DIR = "/root/Workspace/xy/DiT/5script/results/struct_decoder_gpu"


class StructDecoder(nn.Module):
    def __init__(self, in_ch=4, base=64, depth=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU())
        blocks = []
        for _ in range(depth):
            blocks += [nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, base, 3, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.up1 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Conv2d(base, base*4, 1), nn.PixelShuffle(2), nn.SiLU())
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f = self.stem(x); f = self.body(f)
        f = self.up1(f); f = self.up2(f); f = self.up3(f)
        return self.head(f)


def metrics(pred, gt, thr=0.5, eps=1e-6):
    p = (pred > thr).float()
    tp = (p * gt).sum(); fp = (p * (1-gt)).sum(); fn = ((1-p)*gt).sum()
    return ((tp/(tp+fp+fn+eps)).item(), (tp/(tp+fp+eps)).item(), (tp/(tp+fn+eps)).item())


def load_decoder(ckpt_path, base, depth):
    m = StructDecoder(in_ch=4, base=base, depth=depth).to(DEVICE)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def main():
    print("=== 加载数据 (val 2000) ===", flush=True)
    ds = MCCDLatentDataset(
        csv_file="5script/train_full.csv", latent_shards_dir="final_latents",
        img_root=None, canny_root="final_canny_d3", skel_root="final_skeleton_d3",
        image_size=256, load_canny=True, load_skel=True,
        is_train=True, preload=True, load_image=False,
        structure_size=256, num_preload_workers=16)
    n = min(2000, len(ds))
    lat_v = torch.from_numpy(ds._latents[:n].copy()).to(DEVICE)
    sk_v = torch.from_numpy((ds._skels[:n] > 127).astype(np.float32)).unsqueeze(1).to(DEVICE)
    ca_v = torch.from_numpy((ds._cannys[:n] > 127).astype(np.float32)).unsqueeze(1).to(DEVICE)
    print(f"val: {lat_v.shape} skel {sk_v.shape} canny {ca_v.shape}", flush=True)

    results = {}

    # ---- 加载当前 best (base=64, depth=6) ----
    print("\n=== 1. 当前模型 (base=64, depth=6, 348K params) ===", flush=True)
    sk_net, sk_ck = load_decoder(os.path.join(OUT_DIR, "skel_best.pt"), 64, 6)
    ca_net, ca_ck = load_decoder(os.path.join(OUT_DIR, "canny_best.pt"), 64, 6)
    p_sk = sum(x.numel() for x in sk_net.parameters())
    p_ca = sum(x.numel() for x in ca_net.parameters())
    print(f"skel params={p_sk:,} canny params={p_ca:,}", flush=True)

    # 1a. Clean IoU (分 batch 避免显存爆炸)
    sk_ious = []; sk_precs = []; sk_recs = []
    ca_ious = []; ca_precs = []; ca_recs = []
    BS = 100
    with torch.no_grad():
        for bi in range(0, n, BS):
            lat = lat_v[bi:bi+BS]
            sk_p = sk_net(lat).sigmoid()
            ca_p = ca_net(lat).sigmoid()
            si, sp, sr = metrics(sk_p, sk_v[bi:bi+BS])
            ci, cp, cr = metrics(ca_p, ca_v[bi:bi+BS])
            sk_ious.append(si); sk_precs.append(sp); sk_recs.append(sr)
            ca_ious.append(ci); ca_precs.append(cp); ca_recs.append(cr)
    sk_iou = np.mean(sk_ious); sk_prec = np.mean(sk_precs); sk_rec = np.mean(sk_recs)
    ca_iou = np.mean(ca_ious); ca_prec = np.mean(ca_precs); ca_rec = np.mean(ca_recs)
    print(f"Clean: skel IoU={sk_iou:.4f} P={sk_prec:.4f} R={sk_rec:.4f} | "
          f"canny IoU={ca_iou:.4f} P={ca_prec:.4f} R={ca_rec:.4f}", flush=True)

    # 1b. 噪声鲁棒性 (分 batch)
    print("\n--- 噪声鲁棒性 ---", flush=True)
    noise_res = {}
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        sk_ious = []; ca_ious = []
        for bi in range(0, n, BS):
            lat_n = lat_v[bi:bi+BS] + torch.randn_like(lat_v[bi:bi+BS]) * sigma
            with torch.no_grad():
                sk_p = sk_net(lat_n).sigmoid()
                ca_p = ca_net(lat_n).sigmoid()
                si, _, _ = metrics(sk_p, sk_v[bi:bi+BS])
                ci, _, _ = metrics(ca_p, ca_v[bi:bi+BS])
            sk_ious.append(si); ca_ious.append(ci)
        si = float(np.mean(sk_ious)); ci = float(np.mean(ca_ious))
        noise_res[sigma] = {"skel": round(si, 4), "canny": round(ci, 4)}
        print(f"  σ={sigma:.2f}: skel IoU={si:.4f} | canny IoU={ci:.4f}", flush=True)

    # 1c. 梯度方向 + 幅度
    print("\n--- 梯度方向 & 幅度 ---", flush=True)
    lat_g = lat_v[:200].clone().requires_grad_(True)
    sk_logit = sk_net(lat_g)
    sk_loss = F.binary_cross_entropy_with_logits(sk_logit, sk_v[:200])
    g_sk = torch.autograd.grad(sk_loss, lat_g)[0]
    g_sk_norm = g_sk.view(200, -1).norm(dim=1)
    lat_norm = lat_v[:200].view(200, -1).norm(dim=1)
    ratio_sk = g_sk_norm.mean() / (lat_norm.mean() + 1e-8)
    print(f"  skel: grad_norm mean={g_sk_norm.mean():.4f} std={g_sk_norm.std():.4f} | "
          f"latent_norm={lat_norm.mean():.4f} | ratio={ratio_sk:.4f}", flush=True)

    lat_g2 = lat_v[:200].clone().requires_grad_(True)
    ca_logit = ca_net(lat_g2)
    ca_loss = F.binary_cross_entropy_with_logits(ca_logit, ca_v[:200])
    g_ca = torch.autograd.grad(ca_loss, lat_g2)[0]
    g_ca_norm = g_ca.view(200, -1).norm(dim=1)
    ratio_ca = g_ca_norm.mean() / (lat_norm.mean() + 1e-8)
    print(f"  canny: grad_norm mean={g_ca_norm.mean():.4f} std={g_ca_norm.std():.4f} | "
          f"ratio={ratio_ca:.4f}", flush=True)

    # 1d. 显存开销: decoder forward+backward 峰值显存
    print("\n--- 显存开销 (batch=24, 和 DiT 训练一致) ---", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lat_b = torch.randn(24, 4, 32, 32, device=DEVICE, requires_grad=True)
    sk_out = sk_net(lat_b)
    sk_lb = F.binary_cross_entropy_with_logits(sk_out, torch.zeros_like(sk_out))
    sk_lb.backward()
    mem_decoder = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  struct decoder (skel) fwd+bwd: {mem_decoder:.1f} MB", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lat_b2 = torch.randn(24, 4, 32, 32, device=DEVICE, requires_grad=True)
    ca_out = ca_net(lat_b2)
    ca_lb = F.binary_cross_entropy_with_logits(ca_out, torch.zeros_like(ca_out))
    ca_lb.backward()
    mem_canny = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  struct decoder (canny) fwd+bwd: {mem_canny:.1f} MB", flush=True)

    # 对比: VAE decode 的显存 (如果可用)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        from models import VAE
        vae = VAE.from_pretrained("stabilityai/sd-vae-ft-mma").to(DEVICE).eval()
        lat_vae = torch.randn(24, 4, 32, 32, device=DEVICE, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = vae.decode(lat_vae)
        loss_vae = img.mean()
        loss_vae.backward()
        mem_vae = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  VAE decode fwd+bwd:            {mem_vae:.1f} MB", flush=True)
        del vae
    except Exception as e:
        mem_vae = None
        print(f"  VAE: {e!r}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    results["current"] = {
        "config": {"base": 64, "depth": 6, "params": p_sk},
        "clean_iou": {"skel": round(sk_iou, 4), "canny": round(ca_iou, 4)},
        "clean_recall": {"skel": round(sk_rec, 4), "canny": round(ca_rec, 4)},
        "noise_robustness": noise_res,
        "grad_ratio": {"skel": round(ratio_sk.item(), 4), "canny": round(ratio_ca.item(), 4)},
        "grad_norm": {"skel_mean": round(g_sk_norm.mean().item(), 4),
                      "canny_mean": round(g_ca_norm.mean().item(), 4)},
        "memory_MB": {"skel_decoder": round(mem_decoder, 1),
                      "canny_decoder": round(mem_canny, 1),
                      "vae": round(mem_vae, 1) if mem_vae else None},
    }

    # ---- 2. 小模型对比 (base=32, depth=3) ----
    print("\n=== 2. 小模型对比 (base=32, depth=3) ===", flush=True)
    sk_small = StructDecoder(in_ch=4, base=32, depth=3).to(DEVICE)
    ca_small = StructDecoder(in_ch=4, base=32, depth=3).to(DEVICE)
    p_small = sum(x.numel() for x in sk_small.parameters())
    print(f"small params={p_small:,} (vs 348K)", flush=True)

    # 快速训 3 epoch (只 skel, 验证容量是否够)
    opt = torch.optim.AdamW(sk_small.parameters(), lr=2e-3, weight_decay=1e-3)
    pw = torch.tensor(15.0, device=DEVICE)
    for ep in range(3):
        perm = np.random.permutation(min(20000, len(ds)))
        for bi in range(0, len(perm), 256):
            idx = perm[bi:bi+256]
            lat = torch.from_numpy(ds._latents[idx].copy()).to(DEVICE)
            sk = torch.from_numpy((ds._skels[idx] > 127).astype(np.float32)).unsqueeze(1).to(DEVICE)
            logit = sk_small(lat)
            p = logit.sigmoid()
            bce = F.binary_cross_entropy_with_logits(logit, sk, pos_weight=pw)
            eps = 1e-6
            dice = 1.0 - (2.0*(p*sk).sum()+eps)/(p.sum()+sk.sum()+eps)
            loss = bce + dice
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sk_small.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            sk_ious = []
            for bi in range(0, n, BS):
                sk_p = sk_small(lat_v[bi:bi+BS]).sigmoid()
                si, sp, sr = metrics(sk_p, sk_v[bi:bi+BS])
                sk_ious.append(si)
            si = float(np.mean(sk_ious))
        print(f"  ep{ep}: skel IoU={si:.4f} P={sp:.4f} R={sr:.4f}", flush=True)

    # 小模型梯度检查
    lat_g3 = lat_v[:200].clone().requires_grad_(True)
    sk_logit3 = sk_small(lat_g3)
    sk_loss3 = F.binary_cross_entropy_with_logits(sk_logit3, sk_v[:200])
    g3 = torch.autograd.grad(sk_loss3, lat_g3)[0]
    g3_norm = g3.view(200, -1).norm(dim=1)
    ratio3 = g3_norm.mean() / (lat_norm.mean() + 1e-8)

    # 小模型显存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lat_b3 = torch.randn(24, 4, 32, 32, device=DEVICE, requires_grad=True)
    sk_out3 = sk_small(lat_b3)
    sk_lb3 = F.binary_cross_entropy_with_logits(sk_out3, torch.zeros_like(sk_out3))
    sk_lb3.backward()
    mem_small = torch.cuda.max_memory_allocated() / 1024**2

    results["small"] = {
        "config": {"base": 32, "depth": 3, "params": p_small},
        "skel_iou_3ep": round(si, 4),
        "grad_ratio": round(ratio3.item(), 4),
        "memory_MB": round(mem_small, 1),
    }
    print(f"\n小模型: IoU={si:.4f} grad_ratio={ratio3.item():.4f} mem={mem_small:.1f}MB", flush=True)

    # ---- 汇总 ----
    print("\n=== 汇总 ===", flush=True)
    print(f"{'':20s} {'base=64,d=6':>15s} {'base=32,d=3':>15s} {'VAE':>10s}", flush=True)
    print(f"{'params':20s} {p_sk:>15,} {p_small:>15,} {'~80M':>10s}", flush=True)
    print(f"{'skel IoU':20s} {sk_iou:>15.4f} {si:>15.4f} {'N/A':>10s}", flush=True)
    print(f"{'grad ratio':20s} {ratio_sk.item():>15.4f} {ratio3.item():>15.4f} {'N/A':>10s}", flush=True)
    print(f"{'mem (MB, b=24)':20s} {mem_decoder:>15.1f} {mem_small:>15.1f} {mem_vae if mem_vae else 'N/A':>10}", flush=True)

    with open(os.path.join(OUT_DIR, "gradient_health.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {OUT_DIR}/gradient_health.json", flush=True)


if __name__ == "__main__":
    main()