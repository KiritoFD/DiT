"""量化 mode() vs sample() 的差异 —— 判断"不对齐 ref"是否material。

ref: vae.encode(x).latent_dist.sample() * 0.18215
我们(缓存路径): quant_conv(enc(x))[:, :4] * 0.18215  == latent_dist.mode()

对三类图分别测:
  image  — 自然-ish 书法图, 后验应该较窄
  canny  — 二值边缘线稿, 对 VAE 是 OOD, 后验应该很宽
  skel   — 二值骨架线稿 (更细), OOD 更严重
输出: 每个通道的 std (后验宽度) 与 |mode - sample| 的量级。
若 |mode-sample| 远小于 latent 本身的 std, 则 mode 是可接受的近似。
"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.data.vae_io import VAEIO, SCALING
from PIL import Image

N = 8
ROWS = []
import csv
for r in csv.DictReader(open("/root/Workspace/xy/DiT/assets/train_fame-kxl-tj-px60.csv",
                             encoding="utf-8")):
    ROWS.append(r["image_path"])
    if len(ROWS) >= N:
        break

IMG_DIR = "/root/Workspace/xy/DiT"
CANNY_DIR = os.path.join(IMG_DIR, "data/aux/final_canny_base")
SKEL_DIR = os.path.join(IMG_DIR, "data/skel/final_skel3_base")


def load3(path):
    g = Image.open(path).convert("L")
    if g.size != (256, 256):
        g = g.resize((256, 256), Image.LANCZOS)
    a = np.stack([np.asarray(g, np.float32)] * 3, 0) / 127.5 - 1.0
    return a


def img_id(p):
    import re
    return int(re.search(r"(\d+)\.png", p).group(1))


print("VAE loading (CPU, 避免与训练争显存)...")
vae = VAEIO(device="cpu", encode_dtype=torch.float32)
print("ok")

for tag, mk in [("image", lambda p: os.path.join(IMG_DIR, p)),
                ("canny", lambda p: os.path.join(CANNY_DIR, f"{img_id(p)}.png")),
                ("skel",  lambda p: os.path.join(SKEL_DIR, f"{img_id(p)}.png"))]:
    xs, ok = [], 0
    for p in ROWS:
        f = mk(p)
        if not os.path.isfile(f):
            continue
        xs.append(load3(f))
        ok += 1
    if not xs:
        print(f"\n{tag}: 没有可用文件 (查过 {mk(ROWS[0])})")
        continue
    x = torch.from_numpy(np.stack(xs, 0)).float()

    # mode (我们现在的做法)
    with torch.inference_mode():
        h = vae.vae.quant_conv(vae.enc(x))
    h = h.float()
    mu = h[:, : h.shape[1] // 2]
    logvar = h[:, h.shape[1] // 2:]
    mode_lat = mu * SCALING
    std_lat = torch.exp(0.5 * logvar) * SCALING

    # sample (ref 的做法) — 多抽几次估后验
    draws = []
    for s in range(4):
        torch.manual_seed(s)
        with torch.inference_mode():
            z = vae.vae.quant_conv(vae.enc(x))
        z = z.float()
        eps = torch.randn_like(z[:, : z.shape[1] // 2])
        smp = (z[:, : z.shape[1] // 2] + eps * torch.exp(0.5 * z[:, z.shape[1] // 2:])) * SCALING
        draws.append(smp)
    draws = torch.stack(draws)                       # (S,B,4,32,32)
    smp_mean = draws.mean(0)
    smp_std = draws.std(0)

    d_ms = (mode_lat - smp_mean).abs()
    print(f"\n=== {tag}  (n={ok}) ===")
    print(f"  latent 自身 std          = {mode_lat.std().item():.4f}")
    print(f"  后验 std (理论 σ·SCALING)= {std_lat.mean().item():.4f}")
    print(f"  |mode - sample均值|      = {d_ms.mean().item():.4f} "
          f"(max {d_ms.max().item():.4f})")
    print(f"  多次 sample 之间的 std   = {smp_std.mean().item():.4f}")
    ratio = d_ms.mean().item() / max(mode_lat.std().item(), 1e-9)
    print(f"  ratio = |mode-sample| / latent_std = {ratio:.4f} "
          f"-> {'可忽略' if ratio < 0.05 else '需注意' if ratio < 0.2 else '严重'}")
