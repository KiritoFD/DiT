#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dump_eval187_for48.py — 在 48 上为指定 ckpt 导出 eval200_fixed 的**全 187 张**生成图。

为什么要自己写: 48 现成的 experiments/ablation_phase_a/eval_ablation_model.py 只存
前 10 张 (save_top10)，而且用的是 **50 步 Euler**；我们 4090 主线的 eval 用的是
**50 步 Heun (100 NFE)**。海报要和 v54/v66/v68 那些行并排，就必须同一套采样器。

本脚本 = 48 的模型加载逻辑 + **我们主线的 Heun 更新式**（照抄 src/loss/flow_matching.py
的 ddim_sample_loop，heun 分支）:
    ts = linspace(1, 0, steps+1)            # shift=1 → 线性
    v1 = model(x, ts[i]*1000, ...)          # TIME_SCALE=1000（两边一致, 已核对）
    x_e = x + dt*v1
    v2 = model(x_e, ts[i+1]*1000, ...)
    x  = x + dt*0.5*(v1+v2)
无 CFG（等价 cfg=1.0，与主线 eval 的 eval_cfg=1.0 一致), seed=0。

出处对齐:
  模型 init / ckpt["args"] 用法 / EMA 取法  ← 48 的 eval_ablation_model.py
  Heun 更新式 / t*1000 / 线性 schedule      ← 我们 src/loss/flow_matching.py

用法 (在 48 上):
  python tools/dump_eval187_for48.py --ckpt <xxx.pt> --out-dir <dir> --also-gt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

BASE = "/home/ds/Workspace/DiT"
sys.path.insert(0, BASE)
from diffusers.models import AutoencoderKL          # noqa: E402
from src.model import DiT_2Cond_models              # noqa: E402

CSV_STRICT = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
SHARDS_DIR = "/home/ds/Workspace/moyi/data/top10_style23/shards_img"
VAE_PATH = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
ASSETS_DIR = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal"
TIME_SCALE = 1000.0
C = 4


def v_eval(model, x, t, kw):
    """一次速度场评估 (与 flow_matching._v 同语义: tuple 解包 + sigma 通道裁剪)。"""
    v = model(x, t * TIME_SCALE, **kw)
    if isinstance(v, tuple):
        v = v[0]
    if v.shape[1] == 2 * C:
        v = v[:, :C]
    return v


def sample_heun(model, shape, kw, steps, device, seed):
    """我们主线的 Heun (heun_batch=False 分支)。"""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(*shape, device=device, generator=g)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t_i, t_n = ts[i], ts[i + 1]
        dt = (t_n - t_i)
        t_b = t_i.expand(shape[0])
        v1 = v_eval(model, x, t_b, kw)
        x_e = x + dt * v1
        v2 = v_eval(model, x_e, ts[i + 1].expand(shape[0]), kw)
        x = x + dt * 0.5 * (v1 + v2)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv", default=CSV_STRICT)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--also-gt", action="store_true", help="同时导出 187 张 GT")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[dump] ckpt={a.ckpt}\n[dump] out={a.out_dir}\n[dump] device={dev} steps={a.steps} (Heun)")
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    print(f"[dump] ckpt args: model={cfg.model} route={getattr(cfg,'cond_inject_at','')} "
          f"step={ckpt.get('step')}")

    model = DiT_2Cond_models[cfg.model](
        learn_sigma=False,
        norm_type=cfg.norm_type,
        mlp_type=cfg.mlp_type,
        qk_norm=cfg.qk_norm,
        rope=cfg.rope,
        condition_fusion=cfg.condition_fusion,
        cond_fusion_norm=cfg.cond_fusion_norm,
        num_calligraphers=cfg.num_calligraphers,
        num_characters=cfg.num_characters,
        num_script_classes=getattr(cfg, "num_script_classes", 3),
        use_script_cond=getattr(cfg, "use_script_cond", False),
        char_embed_dim=cfg.char_embed_dim,
        callig_embed_dim=cfg.callig_embed_dim,
        script_embed_dim=getattr(cfg, "script_embed_dim", None),
        glyph_inject_layers=cfg.glyph_inject_layers,
        cond_inject_at=getattr(cfg, "cond_inject_at", ""),
        cond_inject_scale=getattr(cfg, "cond_inject_scale", True),
    ).to(dev)
    state = ckpt.get("ema", ckpt.get("model", ckpt.get("delta", ckpt)))
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[dump] load_state_dict: missing={len(missing)} unexpected={len(unexpected)} "
          f"| 可训参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    model.eval()

    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(dev).eval()

    char_remap = {int(k): int(v) for k, v in json.load(
        open(f"{ASSETS_DIR}/char_remap.json", encoding="utf-8")).items()}
    callig_remap = {int(k): int(v) for k, v in json.load(
        open(f"{ASSETS_DIR}/callig_remap.json", encoding="utf-8")).items()}
    font_remap = {int(k): int(v) for k, v in json.load(
        open(f"{ASSETS_DIR}/font_remap.json", encoding="utf-8")).items()}

    id_to_lat = {}
    if a.also_gt:
        import glob
        for sf in sorted(glob.glob(f"{SHARDS_DIR}/shard_*.npz")):
            with np.load(sf) as d:
                lats, ids = d["latents"], d["img_ids"]
                for j, iid in enumerate(ids):
                    id_to_lat[str(iid)] = torch.from_numpy(lats[j]).float()
        print(f"[dump] GT latent 表 {len(id_to_lat)} 条")

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"[dump] csv={a.csv} 行数={len(rows)}")
    os.makedirs(os.path.join(a.out_dir, "pred"), exist_ok=True)
    if a.also_gt:
        os.makedirs(os.path.join(a.out_dir, "gt"), exist_ok=True)

    with torch.no_grad():
        for idx, r in enumerate(rows):
            kw = dict(
                y_callig=torch.tensor([callig_remap.get(int(r["calligrapher_id"]), 0)], device=dev),
                y_script=torch.tensor([font_remap.get(int(r["script_id"]), 0)], device=dev),
                y_char=torch.tensor([char_remap.get(int(r["character_id"]), 0)], device=dev),
            )
            z = sample_heun(model, (1, C, 32, 32), kw, a.steps, dev, a.seed)
            dec = vae.decode(z / 0.18215).sample
            pred = torch.clamp((dec + 1.0) / 2.0, 0, 1)
            p = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(p).save(os.path.join(a.out_dir, "pred", f"{idx:03d}.png"))

            if a.also_gt:
                lat = id_to_lat[str(r["img_id"])].unsqueeze(0).to(dev)
                dg = vae.decode(lat / 0.18215).sample
                gt = torch.clamp((dg + 1.0) / 2.0, 0, 1)
                g = (gt.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(g).save(os.path.join(a.out_dir, "gt", f"{idx:03d}.png"))
            if (idx + 1) % 25 == 0 or idx + 1 == len(rows):
                print(f"  {idx+1}/{len(rows)}", flush=True)

    meta = {"ckpt": a.ckpt, "model": cfg.model, "step": ckpt.get("step"),
            "route": getattr(cfg, "cond_inject_at", ""), "sampler": "heun",
            "steps": a.steps, "nfe": a.steps * 2, "cfg": 1.0, "seed": a.seed,
            "csv": a.csv, "n": len(rows),
            "protocol": "50步Heun / 无CFG / t*1000 / eval200_fixed 187"}
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[dump] ✓ 完成 -> {a.out_dir}")


if __name__ == "__main__":
    main()
