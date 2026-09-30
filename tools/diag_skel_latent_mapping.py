#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_skel_latent_mapping.py — 诊断「骨架 latent <-> 骨架图」这个映射到底可不可用。

回答三个问题:
  Q1. VAE 的 latent->img 往返, 对**骨架**这种极稀疏图保真吗?
      (若解码出来墨量/连通性崩了, 那么任何"在 latent 空间监督骨架"的方案都失去意义)
  Q2. latent MSE 与**像素级**骨架质量单调吗?
      (若单调 -> 可以直接在 latent 空间监督; 若不单调 -> latent MSE 是错的度量)
  Q3. 现成的 probe 能不能直接挂到 LatentSkelStructureLoss 上? 需不需要重训?
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

VAE = "pretrained_models/sd-vae-ft-ema"
PNG_DIR = "data/top10_style23/gt_skel_eval_strict84_png"
SHARD = "data/top10_style23/gt_skel_eval_strict84/shard_00000.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def conn_comp(b):
    """8-连通分量数 (b: bool HxW)。"""
    from scipy.ndimage import label, generate_binary_structure
    return int(label(b, generate_binary_structure(2, 2))[1])


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE, local_files_only=True).eval().to(DEV)
    sc = float(getattr(vae.config, "scaling_factor", 0.18215))
    for p in vae.parameters():
        p.requires_grad_(False)

    def dec(lat):
        with torch.no_grad():
            im = vae.decode(lat.to(DEV).float() / sc).sample
        return ((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()   # (N,256,256) [0,1]

    def enc(img01):
        x = img01.to(DEV).float() * 2 - 1
        if x.dim() == 2:
            x = x[None, None]
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        with torch.no_grad():
            return vae.encode(x.repeat(1, 3, 1, 1)).latent_dist.mean * sc

    # ---------- 数据 ----------
    z = np.load(SHARD)
    L = z["latents"].astype(np.float32)          # (84,4,32,32)
    ids = z["img_ids"].tolist()
    z.close()
    pngs = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(PNG_DIR, "*.png"))}
    print(f"[data] shard latent {L.shape}, png {len(pngs)}, 交集 "
          f"{len(set(ids) & set(pngs))}/{len(ids)}")
    keep = [i for i, k in enumerate(ids) if k in pngs][:64]
    L = L[keep]
    GT = np.stack([np.asarray(Image.open(pngs[ids[i]]).convert("L"), np.uint8) for i in keep])
    ink_t = (GT < 128)                            # (N,256,256) bool
    print(f"[data] 用 {len(keep)} 张; PNG 墨量={ink_t.mean():.4f}, "
          f"平均连通分量={np.mean([conn_comp(m) for m in ink_t]):.1f}")

    # 白底 latent (背景参考点)
    white = torch.ones(1, 1, 256, 256)
    z_bg = enc(white).float().cpu()

    # ---------- Q1: latent -> img 往返 ----------
    print("\n" + "=" * 78)
    print("Q1. VAE latent->img 往返保真度 (骨架)")
    print("=" * 78)
    g = dec(torch.from_numpy(L))                  # (N,256,256)
    ink_p = (g < 0.5).numpy()
    print(f"  存盘 latent 解码 : 墨量 {ink_p.mean():.4f} (目标 {ink_t.mean():.4f}, "
          f"比 {ink_p.mean() / max(ink_t.mean(), 1e-9):.3f})")
    print(f"                     平均连通分量 {np.mean([conn_comp(m) for m in ink_p]):.1f} "
          f"(目标 {np.mean([conn_comp(m) for m in ink_t]):.1f})")
    print(f"                     IoU(解码骨架, 原骨架) = {np.mean([iou(a, b) for a, b in zip(ink_p, ink_t)]):.4f}")
    # 再编码回去, 看 latent 自洽性
    re = enc(g).float().cpu()
    cos = F.cosine_similarity(re.flatten(1), torch.from_numpy(L).flatten(1)).mean()
    print(f"  再编码 vs 存盘 latent: cosine {cos:.4f} | MSE "
          f"{float((re - torch.from_numpy(L)).pow(2).mean()):.5f}")

    # 直接 encode 原 PNG (白底黑线) -> latent
    png01 = torch.from_numpy((GT.astype(np.float32) / 255.0)).unsqueeze(1)
    Lp = enc(png01).float().cpu()
    print(f"  encode(PNG) vs 存盘 latent: cosine "
          f"{F.cosine_similarity(Lp.flatten(1), torch.from_numpy(L).flatten(1)).mean():.4f} | MSE "
          f"{float((Lp - torch.from_numpy(L)).pow(2).mean()):.5f}")

    # ---------- Q2: latent MSE 与像素 IoU 是否单调 ----------
    print("\n" + "=" * 78)
    print("Q2. latent MSE  vs  像素级骨架质量 (单调性)")
    print("=" * 78)
    print("  扰动方式: 把骨架 latent 往白底 latent 方向插值 (模拟'墨量塌陷'), 加噪 (模拟'糊')")
    print(f"  {'扰动':<26}{'latentMSE':>11}{'latentCos':>11}{'像素IoU':>10}"
          f"{'解码墨量':>10}{'连通分量':>9}")
    rows = []
    Lt = torch.from_numpy(L)
    zbg = z_bg.expand_as(Lt)
    for a in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        Lx = (1 - a) * Lt + a * zbg
        mse = float((Lx - Lt).pow(2).mean())
        cs = float(F.cosine_similarity(Lx.flatten(1), Lt.flatten(1)).mean())
        d = dec(Lx)
        m = (d < 0.5).numpy()
        rows.append(("插值->白底 a=%.2f" % a, mse, cs, float(np.mean([iou(x, y) for x, y in zip(m, ink_t)])),
                     float(m.mean()), float(np.mean([conn_comp(q) for q in m]))))
    for s in [0.1, 0.3, 0.6, 1.0]:
        Lx = Lt + torch.randn_like(Lt) * (s * float(Lt.std()))
        mse = float((Lx - Lt).pow(2).mean())
        cs = float(F.cosine_similarity(Lx.flatten(1), Lt.flatten(1)).mean())
        d = dec(Lx)
        m = (d < 0.5).numpy()
        rows.append(("加噪 s=%.2f" % s, mse, cs, float(np.mean([iou(x, y) for x, y in zip(m, ink_t)])),
                     float(m.mean()), float(np.mean([conn_comp(q) for q in m]))))
    for r in rows:
        print(f"  {r[0]:<26}{r[1]:>11.5f}{r[2]:>11.4f}{r[3]:>10.4f}{r[4]:>10.4f}{r[5]:>9.1f}")

    try:
        from scipy.stats import spearmanr
        ms = [r[1] for r in rows]
        ious = [r[3] for r in rows]
        rho = spearmanr(ms, ious).statistic
        print(f"\n  Spearman(latentMSE, 像素IoU) = {rho:+.3f}  "
              f"(<-1 表示越'差'越像, 越接近 -1 越单调可用)")
    except Exception as e:
        print("  spearman 失败:", e)

    # ---------- Q3: 现成 probe 是否可用 ----------
    print("\n" + "=" * 78)
    print("Q3. 现成 probe 能否挂 LatentSkelStructureLoss")
    print("=" * 78)
    ck = "assets/results/latent_struct_probe.pt"
    if not os.path.exists(ck):
        print("  没有 probe 检查点")
        return
    d = torch.load(ck, map_location="cpu", weights_only=False)
    sd = d["model"]
    print(f"  文件: {ck}")
    print(f"  args: {d.get('args')}")
    print(f"  metrics: {d.get('metrics')}")
    # 推断结构
    import re as _re
    stem_w = None
    head_out = None
    for k, v in sd.items():
        if k.endswith("stem.weight"):
            stem_w = tuple(v.shape)
        if _re.match(r"head\.\d+\.weight", k) and v.dim() == 4:
            head_out = tuple(v.shape)
    print(f"  stem.weight={stem_w}  head 输出层={head_out}")
    print(f"  -> 这是 **LatentStructureProbe**(4ch latent -> 2ch 像素 logits, width 32)")
    print(f"     而 LatentSkelStructureLoss 要求 probe 输出 4ch (实例骨架 latent)")
    print(f"     -> **类不兼容, 不能直接用**。")
    # 试一下它输出的东西长什么样
    try:
        from src.train.latent_structure import LatentStructureProbe, LatentSkelProbe
        p = LatentStructureProbe(in_channels=4, width=32, depth=2)
        miss = p.load_state_dict(sd, strict=False)
        print(f"     LatentStructureProbe 载入: missing={len(miss.missing_keys)} "
              f"unexpected={len(miss.unexpected_keys)}")
        p.eval()
        with torch.no_grad():
            o = p(torch.from_numpy(L[:8]))
        print(f"     probe(真骨架 latent) 输出 shape={tuple(o.shape)} "
              f"sigmoid均值={float(o.sigmoid().mean()):.4f} "
              f"每通道={[round(float(o.sigmoid()[:, c].mean()), 4) for c in range(o.shape[1])]}")
        print(f"     (2ch = [canny, skeleton] 像素 logits; 均值 0.5 附近 = 没学到判别性)")
    except Exception as e:
        print("     探针前向失败:", repr(e))
    # LatentSkelProbe 是否存在
    print(f"\n  LatentSkelProbe 检查点搜索:")
    for pat in ["assets/structure_probes/**/*.pt", "assets/**/*skel_probe*.pt",
                "assets/results/*skel*.pt"]:
        hit = glob.glob(pat, recursive=True)
        print(f"     {pat:<40} -> {hit if hit else '无'}")


if __name__ == "__main__":
    main()
