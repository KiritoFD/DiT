#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_probe_sensitivity.py — probe 作为结构 loss 到底能不能"看见"墨量塌陷?

`LatentSkelStructureLoss` 的形式是 MSE(probe(pred_xstart), skel_latent_gt)。
它能不能救塌陷, 取决于一个唯一的问题:

    **把输入 latent 的墨量逐步抹掉, probe 的输出跟着变吗?**

  - 若 probe 输出纹丝不动 (还能"脑补"出完整骨架) -> loss 对塌陷不敏感 -> 挂上去白挂。
  - 若 probe 输出同步退化 -> loss 能感知 -> 可用, 剩下的只是权重 w 的问题。

同时给出对照: 原始 latent MSE 在同一扰动下的表现 (看 probe 是否提供了**额外**信号)。
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
PNG_DIR = "data/top10_style23/gt_skel_eval_strict84_png"


def main():
    from diffusers.models import AutoencoderKL
    from src.train.latent_structure import LatentSkelProbe
    from scipy.ndimage import label, generate_binary_structure

    probe_path = sys.argv[1] if len(sys.argv) > 1 else \
        "assets/structure_probes/latent_skel_probe_v1_mse/best.pt"
    ck = torch.load(probe_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    print(f"[probe] {probe_path}")
    print(f"[probe] args={a}  metrics={ck.get('metrics')}")
    probe = LatentSkelProbe(in_channels=a["in_channels"], out_channels=a["out_channels"],
                            width=a["width"], depth=a["depth"]).eval().to(DEV)
    ms = probe.load_state_dict(ck["model"], strict=False)
    print(f"[probe] 载入 missing={len(ms.missing_keys)} unexpected={len(ms.unexpected_keys)}")
    for p in probe.parameters():
        p.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(VAE, local_files_only=True).eval().to(DEV)
    sc = float(getattr(vae.config, "scaling_factor", 0.18215))

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

    # ── 取一批 (图像 latent, 骨架 latent) 配对 ──
    def idx(d):
        o = {}
        for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            z = np.load(sp)
            for j, iid in enumerate(z["img_ids"]):
                o[int(iid)] = (sp, j)
            z.close()
        return o
    ii, si = idx(IMG_SH), idx(SKEL_SH)
    common = sorted(set(ii) & set(si))
    print(f"[data] 配对 {len(common)}  (img {len(ii)} / skel {len(si)})")
    rng = np.random.RandomState(0)
    pick = [common[k] for k in rng.choice(len(common), 48, replace=False)]

    def get(m, iid):
        sp, j = m[iid]
        with np.load(sp) as z:
            return z["latents"][j].astype(np.float32)
    X = torch.from_numpy(np.stack([get(ii, i) for i in pick]))
    Y = torch.from_numpy(np.stack([get(si, i) for i in pick]))

    z_bg = enc(torch.ones(1, 1, 256, 256)).float().cpu()
    st = generate_binary_structure(2, 2)

    # ── 扰动: 图像 latent 往白底插值 = 精确模拟"墨量塌陷" ──
    print("\n" + "=" * 92)
    print("把**图像** latent 往白底插值 (模拟墨量塌陷), 看 probe 输出与各 loss 怎么变")
    print("=" * 92)
    GT = (dec(Y) < 0.5)                       # 真骨架 (由 GT 骨架 latent 解码, 上限口径 IoU=0.956)
    hdr = (f"  {'α':>6}{'rawMSE(x,x0)':>14}{'probeLoss':>11}"
           f"{'probe解码墨量':>14}{'IoU(probe,真骨架)':>18}{'图像墨量':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for al in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        Xa = (1 - al) * X + al * z_bg.expand_as(X)
        raw = float((Xa - X).pow(2).mean())
        ps = probe(Xa.to(DEV)).float().cpu()
        pl = float(F.mse_loss(ps, Y))
        mp = (dec(ps) < 0.5)
        ink_p = float(mp.float().mean())
        iou_s = float(((mp & GT).sum().float() / (mp | GT).sum().clamp_min(1).float()))
        di = dec(Xa)
        print(f"  {al:>6.2f}{raw:>14.5f}{pl:>11.5f}{ink_p:>14.4f}{iou_s:>18.4f}"
              f"{float((di < 0.5).float().mean()):>10.4f}")
    print(f"  (真骨架参考: 墨量 {float(GT.float().mean()):.4f})")

    # ── 关键量: probe 对塌陷的"敏感度" ──
    print("\n" + "=" * 92)
    print("敏感度 = (loss(α=1) - loss(α=0)) / loss(α=0)   —— 越大说明 loss 越能看见塌陷")
    print("=" * 92)
    for name, fn in [("rawMSE(图像latent)", lambda Xa: float((Xa - X).pow(2).mean())),
                     ("probeLoss(骨架latent)", lambda Xa: float(F.mse_loss(probe(Xa.to(DEV)).float().cpu(), Y)))]:
        l0, l1 = fn(X), fn((1 - 1.0) * X + z_bg.expand_as(X))
        print(f"  {name:<26} loss(α=0)={l0:.5f}  loss(α=1)={l1:.5f}  敏感度 {l1/max(l0,1e-9):.2f}x")
    # probe 解码墨量随 α 的衰减 = probe 是否"脑补"
    for al in [0.0, 0.3, 0.5, 1.0]:
        Xa = (1 - al) * X + al * z_bg.expand_as(X)
        m = (dec(probe(Xa.to(DEV)).float().cpu()) < 0.5).float().mean()
        print(f"  probe 解码墨量 @ α={al:.2f}: {float(m):.4f}")


if __name__ == "__main__":
    main()
