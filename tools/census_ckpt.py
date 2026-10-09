#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""census_ckpt.py — 通用 DiT checkpoint 普查：高频保真 + 墨量均值化（逐字回归斜率）

用途（v74@85k vs v72@85k 的判定）:
  * HF×GT        —— 越大越好（现在 v72 是 0.51~0.67）
  * ink 斜率/r   —— 斜率越接近 1、r 越高 = 墨量越"跟着字形走"（v72 现在斜率 0.55 / r 0.53）

用法:
  python tools/census_ckpt.py --ckpt <path> --config <cfg.json> --vae-path <vae> \
      --step 85000 --n 40 --out experiments/census_v74_85k
"""

import argparse
import copy
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"


class A:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def img_stats(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    ink = float((a < 128).mean())
    b = a / 255.0
    gx = np.zeros_like(b); gy = np.zeros_like(b)
    gx[:, 1:-1] = b[:, 2:] - b[:, :-2]
    gy[1:-1, :] = b[2:, :] - b[:-2, :]
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return ink, float(np.sqrt(gx ** 2 + gy ** 2).mean()), float(F[r > 0.5].sum() / max(F.sum(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0")

    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
    idx = [round(i * (len(rows) - 1) / (a.n - 1)) for i in range(a.n)]
    csvp = os.path.join(a.out, f"census{a.n}.csv")
    with open(csvp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows([rows[i] for i in idx])
    setname = "census"

    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg.update({
        "eval_csv": EVAL_CSV, "in_mem_eval_sets": f"{setname}:{csvp}:{a.n}",
        "triple_table_prefix": TABLES + "/",
        "char_remap_json": f"{TABLES}/char_remap.json",
        "callig_remap_json": f"{TABLES}/callig_remap.json",
        "font_remap_json": f"{TABLES}/font_remap.json",
        "vae_path": a.vae_path, "eval_vae_path": a.vae_path,
        "eval_cfg": a.cfg, "sde_gamma": a.gamma, "eval_steps": 50,
        "in_mem_eval_batch": 16, "in_mem_eval_vae_batch": 16,
        "in_mem_eval_save_samples": True,
    })
    model = DiT_2Cond_models[cfg.get("model", "DiT-2Cond-Sp/2")](
        learn_sigma=False, norm_type=cfg.get("norm_type", "layer"), mlp_type=cfg.get("mlp_type", "gelu"),
        qk_norm=cfg.get("qk_norm", 0), rope=cfg.get("rope", 0),
        condition_fusion=cfg.get("condition_fusion", "factorized_cat"),
        cond_fusion_norm=cfg.get("cond_fusion_norm", "split"),
        num_calligraphers=cfg.get("num_calligraphers", 10), num_characters=cfg.get("num_characters", 4690),
        num_script_classes=cfg.get("num_script_classes", 3), use_script_cond=cfg.get("use_script_cond", True),
        char_embed_dim=cfg.get("char_embed_dim", 256), callig_embed_dim=cfg.get("callig_embed_dim", 32),
        script_embed_dim=cfg.get("script_embed_dim", 16),
        glyph_inject_layers=cfg.get("glyph_inject_layers", 0),
        cond_inject_at=cfg.get("cond_inject_at", "2,4,5,6"),
        cond_inject_scale=cfg.get("cond_inject_scale", True)).to(dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st = ck.get("ema") or ck.get("model") or ck
    if hasattr(st, "state_dict"):
        st = st.state_dict()
    st = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in st.items()}
    m, u = model.load_state_dict(st, strict=False)
    print(f"[ckpt] missing={m} unexpected={u}", flush=True)
    model.eval()

    res = run_in_mem_eval(model, A(cfg), step=a.step, device=dev, results_dir=a.out, logger=lambda *x: None)
    e = (res or {}).get(setname, {}) or {}
    d = os.path.join(a.out, "eval_samples_ctrl", f"step{a.step:07d}", setname)
    xs, ys, gt_hf, gen_hf, gts, gens = [], [], [], [], [], []
    for f in sorted(os.listdir(d)):
        if f.startswith("gt") and f.endswith(".png"):
            i = f[2:-4]
            gp, op = os.path.join(d, f), os.path.join(d, f"g{i}.png")
            if not os.path.exists(op):
                continue
            ti, _, thf = img_stats(gp)
            oi, _, ohf = img_stats(op)
            xs.append(ti); ys.append(oi); gt_hf.append(thf); gen_hf.append(ohf)
            gts.append(ti); gens.append(oi)
    xs = np.array(xs); ys = np.array(ys)
    out = {
        "ckpt": a.ckpt, "step": a.step, "n": len(xs), "cfg": a.cfg, "gamma": a.gamma,
        "ssim": e.get("ssim"), "lpips": e.get("lpips"), "ink_iou": e.get("ink_iou"), "frag": e.get("frag"),
        "gt_ink_mean": round(float(xs.mean()), 4), "gen_ink_mean": round(float(ys.mean()), 4),
        "hf_ratio": round(float(np.mean(gen_hf) / max(np.mean(gt_hf), 1e-9)), 4),
        "ink_ratio": round(float(ys.mean() / max(xs.mean(), 1e-9)), 4),
    }
    if len(xs) >= 3:
        slope, _ = np.polyfit(xs, ys, 1)
        out["ink_slope"] = round(float(slope), 4)
        out["ink_r"] = round(float(np.corrcoef(xs, ys)[0, 1]), 4)
    json.dump(out, open(os.path.join(a.out, "census_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"-> {a.out}/census_summary.json")


if __name__ == "__main__":
    main()
