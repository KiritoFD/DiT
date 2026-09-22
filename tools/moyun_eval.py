#!/usr/bin/env python
"""moyun 复现的评测 + poster（CPU）。

## 与原版 generate.py 的对应
原版（`ref/moyi/moyun/generate.py`）:
    samples, _ = samples.chunk(2, dim=0)      # 去掉 CFG 的 null 分支
    samples = samples[:, 0:4, :, :]           # 只取 image latent，8 个 aux 通道丢弃
    vae.decode(samples / 0.18215)

本脚本一致：DDIM 采样 12ch -> 取前 4ch -> VAE decode。

## 为什么不用 batch_eval.py
那个是我们模型的专用评测（flow + 2cond），不认 moyun 的
（12ch + DDPM + LabelEmbedder 三表 + learn_sigma）。

用法:
  PYTHONPATH=ref/moyi CUDA_VISIBLE_DEVICES= python tools/moyun_eval.py \
      --ckpt assets/results/moyun_repro/moyun_0015000.pt \
      --csv assets/eval_v13_seen_fixed.csv --n 20 --device cpu
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
sys.path.insert(0, os.path.join(_ROOT, "ref", "moyi", "moyun"))
os.chdir(_ROOT)

from dataset_moyun import MultiLabelNestedDataset  # noqa: E402
from moyun_2 import DiT_models  # noqa: E402
from utils.REPA_diffusion import create_diffusion  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default="assets/eval_v13_seen_fixed.csv")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--num-classes", type=int, default=9100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    p.add_argument("--save-samples", action="store_true")
    return p.parse_args()


def main():
    a = get_args()
    dev = torch.device(a.device)
    out = a.out or os.path.join(
        os.path.dirname(a.ckpt), f"eval_{os.path.basename(a.ckpt)[:-3]}")
    os.makedirs(out, exist_ok=True)
    print(f"[moyun-eval] ckpt={a.ckpt}\n             out={out}", flush=True)

    # ── 数据（复用训练时的 dataset，保证 id 对齐）──────────────────
    ds = MultiLabelNestedDataset(
        csv_file=a.csv, img_shards="data/50k/shards_img",
        edge_shards="data/50k/shards_aux_canny",
        skel_shards="data/50k/shards_std_fixed", num_classes=a.num_classes)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))[:len(ds)]
    n = min(a.n, len(ds))
    print(f"[moyun-eval] {len(ds)} 条，评 {n} 条", flush=True)

    # ── 模型 ────────────────────────────────────────────────────────
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("model")
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model = DiT_models["moyun-12channel-B"](
        input_size=32, num_classes=a.num_classes,
        learn_sigma=True, if_rope=False)
    ms, us = model.load_state_dict(sd, strict=False)
    if ms:
        print(f"  ⚠ missing={len(ms)}: {list(ms)[:3]}")
    model = model.to(dev).eval()

    diffusion = create_diffusion(timestep_respacing="", learn_sigma=True,
                                 device=str(dev))

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "data/pretrained/pretrained_models/sd-vae-ft-ema").to(dev).eval()
    sf = 0.18215

    # ── 逐样本采样 ──────────────────────────────────────────────────
    gs, gts = [], []
    for i in range(n):
        image, edge, skel, y, stroke, *_ = ds[i]
        y = y.squeeze(-1).unsqueeze(0).to(dev)          # (1,3)
        stroke = stroke.unsqueeze(0).to(dev)
        # 12ch 全从噪声生成（原版就是这么做的 —— 没有结构条件）
        # ⚠ forward_with_cfg 内部 
        #   -> 输入 batch 必须是**真实 batch 的 2 倍**（否则 half 为空 -> ZeroDivision）
        #   所以这里用 batch=2（同一样本复制两份），取前一半的结果
        shape = (2, 12, 32, 32)
        with torch.no_grad():
            _y2 = torch.cat([y, y], dim=0)
            _s2 = torch.cat([stroke, stroke], dim=0)
            lat = diffusion.ddim_sample(
                model, shape, {"y": _y2, "stroke": _s2},
                steps=a.steps, cfg_scale=a.cfg)
        lat = lat[:1]
        rec = vae.decode(lat[:, :4].to(dev) / sf).sample.float().cpu()
        gimg = ((rec[0].detach().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
        gs.append((gimg * 255).astype(np.uint8))
        # GT: **直接读原图**（和 src/eval/inference.py:509-512 一致）
        #   ⚠ 不要 decode latent —— csv 的 image_path 就是 256 的原图，
        #     decode 一遍既慢又引入 VAE 误差
        _p = rows[i]["image_path"]
        if not os.path.isabs(_p):
            _p = os.path.join(_ROOT, _p)
        gts.append(np.asarray(Image.open(_p).convert("RGB").resize((256, 256)),
                              dtype=np.uint8))
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n}", flush=True)

    # ── 指标（ssim / lpips）──────────────────────────────────────────
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        _ssim = None
    ssims = []
    for g, t in zip(gs, gts):
        if _ssim is not None:
            ssims.append(_ssim(t, g, channel_axis=2, data_range=255))
    if ssims:
        print(f"\n  ssim = {np.mean(ssims):.4f}  (med={np.median(ssims):.4f})",
              flush=True)

    try:
        import lpips as _lp
        fn = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
        vs = []
        for g, t in zip(gs, gts):
            x = torch.from_numpy(g).float().permute(2, 0, 1)[None].to(dev) / 127.5 - 1
            y2 = torch.from_numpy(t).float().permute(2, 0, 1)[None].to(dev) / 127.5 - 1
            with torch.no_grad():
                vs.append(float(fn(x, y2)))
        print(f"  lpips = {np.mean(vs):.4f}", flush=True)
    except Exception as e:
        print(f"  lpips 不可用: {str(e)[:60]}")

    # ── 落盘 + poster ───────────────────────────────────────────────
    if a.save_samples:
        sd_dir = os.path.join(out, "samples")
        os.makedirs(sd_dir, exist_ok=True)
        for i, (g, t) in enumerate(zip(gs, gts)):
            Image.fromarray(g).save(os.path.join(sd_dir, f"g{i}.png"))
            Image.fromarray(t).save(os.path.join(sd_dir, f"gt{i}.png"))
        print(f"  samples -> {sd_dir}")

        # poster: 上排生成、下排 GT
        cols = min(n, 10)
        W = 128
        canvas = Image.new("RGB", (W * cols, W * 2 + 24), "white")
        for i in range(cols):
            canvas.paste(Image.fromarray(gs[i]).resize((W - 4, W - 4)),
                         (i * W + 2, 20))
            canvas.paste(Image.fromarray(gts[i]).resize((W - 4, W - 4)),
                         (i * W + 2, W + 24))
        p = os.path.join(out, "poster.png")
        canvas.save(p)
        print(f"  poster -> {p}")

    with open(os.path.join(out, "eval_summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "n", "ssim_mean", "ssim_med", "steps", "cfg"])
        w.writerow([a.ckpt, len(gs),
                    round(float(np.mean(ssims)), 4) if ssims else "",
                    round(float(np.median(ssims)), 4) if ssims else "",
                    a.steps, a.cfg])
    print(f"\n  -> {os.path.join(out, 'eval_summary.csv')}")


if __name__ == "__main__":
    main()
