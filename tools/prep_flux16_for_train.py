"""换 16ch VAE 的两步数据准备 (一次跑完, 幂等):

① shard 常量平移: 把 (z-shift)*sf 转成 raw*sf —— 只需 +shift*sf = +0.04185。
   为什么: 仓库现有解码约定是 `z / sf` (= raw), 而 Flux 解码器正好要 raw latent
   (官方 pipeline: latent = pred/sf + shift 再 decode)。把 shift 折进存储, 就**不用改任何解码代码**。
② 存一份可直接 from_pretrained 的 Flux VAE: 真 Flux AE 没有 quant_conv/post_quant_conv,
   而 diffusers 0.27 的 AutoencoderKL 会自建 -> from_pretrained 报缺 key。这里用恒等
   矩阵填好后 save_pretrained, 于是 load_eval_vae(path) 能直接加载, 评测侧零改动。

用法: python tools/prep_flux16_for_train.py
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from flux_ae import DEFAULT_DIR, load_flux_ae  # noqa: E402

SF, SHIFT = 0.3611, 0.1159
DELTA = SHIFT * SF                      # 0.041850...
DIRS = ["exp-std/data/shards_img_flux16", "exp-std/data/shards_std_flux16",
        "exp-std/data/shards_gtskel_flux16",
        "exp-std/data/shards_std_flux16_fixed_eval200"]
MARK = "exp-std/data/.flux16_shift_folded"

# ── ① shard 平移 ────────────────────────────────────────────────────
if os.path.exists(MARK):
    print(f"[skip] {MARK} 已存在 -> shard 已平移过 (幂等)")
else:
    for d in DIRS:
        fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
        if not fs:
            print(f"[warn] {d} 没有 shard, 跳过")
            continue
        t0 = time.time()
        for f in fs:
            z = np.load(f)
            lat = z["latents"].astype(np.float32) + DELTA
            tmp = f + ".tmp.npz"
            np.savez_compressed(tmp, latents=lat.astype(np.float16), img_ids=z["img_ids"])
            os.replace(tmp, f)
        print(f"[fold] {d}: {len(fs)} shard +{DELTA:.5f}  ({time.time() - t0:.0f}s)")
    with open(MARK, "w") as fh:
        fh.write(f"delta={DELTA}\ndirs={DIRS}\n")
    print(f"[mark] 写入 {MARK}")

# 校验: 取一个 shard 反算, 确认解码约定一致
z = np.load(sorted(glob.glob(DIRS[0] + "/shard_*.npz"))[0])
print(f"[check] {DIRS[0]} latents{z['latents'].shape} {z['latents'].dtype} "
      f"mean={float(z['latents'].astype(np.float32).mean()):.4f}  (应≈0 附近, 与 sd 4ch 同量级)")

# ── ② 存可加载的 Flux VAE ───────────────────────────────────────────
OUT = "data/pretrained/flux_vae_eval"
if os.path.isdir(OUT) and os.path.exists(os.path.join(OUT, "config.json")):
    print(f"[skip] {OUT} 已存在")
else:
    vae, sf, shift = load_flux_ae("cpu", diff_dir=DEFAULT_DIR)
    os.makedirs(OUT, exist_ok=True)
    vae.save_pretrained(OUT)
    cfgp = os.path.join(OUT, "config.json")
    c = json.load(open(cfgp, encoding="utf-8"))
    c.update({"scaling_factor": sf, "shift_factor": shift, "latent_channels": 16})
    json.dump(c, open(cfgp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[vae] 写出 {OUT}: {sorted(os.listdir(OUT))}")
    print(f"      scaling={sf} shift={shift} latent_channels=16")

# 自检: from_pretrained 能否加载 + 往返 PSNR
from diffusers import AutoencoderKL  # noqa: E402
v = AutoencoderKL.from_pretrained(OUT).eval()
with th.no_grad():
    x = th.randn(2, 3, 256, 256).clamp(-1, 1)
    z = v.encode(x).latent_dist.mode()
    d = v.decode(z).sample
    m = float(((d.clamp(-1, 1) - x) ** 2).mean())
print(f"[selfcheck] from_pretrained ok; 随机图往返 MSE={m:.4f} "
      f"(只验通路能跑通, 不是质量指标)")
print("[done] 16ch 准备完成")
