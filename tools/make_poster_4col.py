#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_poster_4col.py — 生成 6 列对比海报：输入std / GT / v68γ0 / v68γ0.5 / v72γ0 / v72γ0.5

目的：不看数字，直接看图判断
  (1) 我们（v72/Calli-VAE）生成的字 vs v68，到底谁好；
  (2) v68 的 SDE "free lunch"（γ0→γ0.5，SSIM +0.02）在视觉上有没有坏处（糊/胖/失细节）。

严格对照：同一批 12 个字、同一书家书体、**同一初始噪声（seed=0 + 模块级缓存）**、
同一 CFG=0.7、同一 heun-50、同一 VAE（各自的）。
"""

import argparse
import csv
import json
import os
import sys

import torch

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"
SD_VAE = f"{ROOT}/data/pretrained/pretrained_models/sd-vae-ft-ema"
CALLI_VAE = f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"

V68_CKPT = (f"{ROOT}/assets/runs/v68_family/v68_aug_sp_c2ot/"
            "20261006-192021-v68-aug-sp-c2ot/checkpoints/0200000.pt")
V68_CFG = f"{ROOT}/src/train/configs/v68_aug_sp_c2ot.json"
V72_CKPT = (f"{ROOT}/assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/"
            "20261009-142151-v72-callivae-kl1e6-s12500-frozen-sp-c2ot/checkpoints/0130000.pt")
V72_CFG = f"{ROOT}/src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json"

COLS = [
    ("v68_g0.0", V68_CKPT, V68_CFG, SD_VAE, 200000, 0.7, 0.0),
    ("v68_g0.5", V68_CKPT, V68_CFG, SD_VAE, 200000, 0.7, 0.5),
    ("v72_g0.0", V72_CKPT, V72_CFG, CALLI_VAE, 130000, 0.7, 0.0),
    ("v72_g0.5", V72_CKPT, V72_CFG, CALLI_VAE, 130000, 0.7, 0.5),
]


class ArgsObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def pick_rows(n=12):
    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
    total = len(rows)
    idx = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    sel = [rows[i] for i in idx]
    out = os.path.join(ROOT, "exp-std/csv/poster12.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sel)
    print(f"[pick] {n}/{total} rows -> {out}")
    for i, r in zip(idx, sel):
        print(f"   #{i:>3} {r.get('character','?')} {r.get('script','?')} {r.get('calligrapher','?')}")
    return out, len(sel)


def build(cfg, device):
    return DiT_2Cond_models[cfg.get("model", "DiT-2Cond-Sp/2")](
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
        cond_inject_scale=cfg.get("cond_inject_scale", True),
    ).to(device)


def load_into(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck.get("ema") or ck.get("model") or ck
    if hasattr(st, "state_dict"):
        st = st.state_dict()
    st = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in st.items()}
    m, u = model.load_state_dict(st, strict=False)
    print(f"   [load] {os.path.basename(path)} missing={m} unexpected={u}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/poster_4col")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cell", type=int, default=176)
    ap.add_argument("--only", default="all", help="只跑某一模型: v68 / v72 / all （一进程一模型，避免 VAE 缓存串味）")
    ap.add_argument("--compose-only", action="store_true", help="只拼海报，不重新采样")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, n = pick_rows(args.n)
    sets = f"poster:{ROOT}/exp-std/csv/poster12.csv:{n}"

    # 按 ckpt 分组，模型只建一次
    cols = [c for c in COLS if args.only == "all" or c[0].startswith(args.only)]
    print(f"[only={args.only}] 本次将采样 {len(cols)} 列: {[c[0] for c in cols]}", flush=True)
    by_ckpt = {}
    if not args.compose_only:
        for label, ckpt, cfgp, vae, step, cfgv, gam in cols:
            by_ckpt.setdefault((ckpt, cfgp, vae, step), []).append((label, cfgv, gam))

    for (ckpt, cfgp, vae, step), settings in by_ckpt.items():
        cfg = json.load(open(cfgp, encoding="utf-8"))
        cfg.update({
            "eval_csv": EVAL_CSV, "in_mem_eval_sets": sets,
            "triple_table_prefix": TABLES + "/",
            "char_remap_json": f"{TABLES}/char_remap.json",
            "callig_remap_json": f"{TABLES}/callig_remap.json",
            "font_remap_json": f"{TABLES}/font_remap.json",
            "vae_path": vae, "eval_vae_path": vae, "calli_decode_noise": False,
            "eval_steps": 50, "in_mem_eval_batch": 16, "in_mem_eval_vae_batch": 16,
        })
        print(f"\n=== model {os.path.basename(os.path.dirname(os.path.dirname(ckpt)))} "
              f"{os.path.basename(os.path.dirname(ckpt))} ===", flush=True)
        model = build(cfg, device)
        load_into(model, ckpt)
        model.eval()
        for label, cfgv, gam in settings:
            c = dict(cfg); c["eval_cfg"] = cfgv; c["sde_gamma"] = gam
            c["in_mem_eval_save_samples"] = True
            d = os.path.join(args.out, label)
            print(f"\n>>> sample [{label}] cfg={cfgv} gamma={gam} -> {d}", flush=True)
            res = run_in_mem_eval(model, ArgsObject(c), step=step, device=device, results_dir=d, logger=print)
            e = (res or {}).get("eval200fix", {}) or {}
            print(f"    {label}: ssim={e.get('ssim')} lpips={e.get('lpips')} frag={e.get('frag')}", flush=True)

    # ---------- 组装海报 ----------
    from PIL import Image, ImageDraw
    from skimage.metrics import structural_similarity as ssim_fn
    import numpy as np

    def cell_path(label, i, kind="g"):
        p = os.path.join(args.out, label, "eval_samples_ctrl", f"step{200000 if label.startswith('v68') else 130000:07d}",
                         "poster", f"{kind}{i}.png")
        return p if os.path.exists(p) else None

    def std_path(i):
        p = os.path.join(args.out, "v68_g0.0", "eval_samples_ctrl", "poster_input_g", f"g{i}.png")
        return p if os.path.exists(p) else None

    cs = args.cell
    labels = ["std", "GT"] + [c[0] for c in COLS]
    W = cs * len(labels)
    H = cs * n + 34
    poster = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(poster)
    for j, lab in enumerate(labels):
        dr.text((j * cs + 8, 10), lab, fill="black")
    dr.line([(0, 32), (W, 32)], fill="gray")

    # GT 取自任意一次运行（同一批）
    gt0 = cell_path("v68_g0.0", 0, "gt")
    gt_dir = os.path.dirname(gt0) if gt0 else None

    per_col = {lab: [] for lab in labels[2:]}
    for i in range(n):
        y = 34 + i * cs
        # 输入标准字
        sp = std_path(i)
        if sp:
            poster.paste(Image.open(sp).convert("RGB").resize((cs, cs)), (0, y))
        # GT
        gp = os.path.join(gt_dir, f"gt{i}.png") if gt_dir else None
        gt_arr = None
        if gp and os.path.exists(gp):
            gimg = Image.open(gp).convert("RGB").resize((cs, cs))
            poster.paste(gimg, (cs, y))
            gt_arr = np.asarray(gimg.convert("L"), dtype=np.float32)
        # 四列生成
        for j, (lab, *_rest) in enumerate(COLS):
            p = cell_path(lab, i, "g")
            if not p:
                continue
            gimg = Image.open(p).convert("RGB").resize((cs, cs))
            poster.paste(gimg, ((2 + j) * cs, y))
            if gt_arr is not None:
                a = np.asarray(gimg.convert("L"), dtype=np.float32)
                s = ssim_fn(gt_arr, a, data_range=255.0)
                per_col[lab].append(s)
                dr.text(((2 + j) * cs + 4, y + cs - 14), f"{s:.3f}", fill="red")
    out_png = os.path.join(args.out, "poster_v68_vs_v72.png")
    poster.save(out_png)
    print(f"\n[poster] {out_png}  ({poster.size[0]}x{poster.size[1]})")

    print("\n== 这 12 个字上的逐列平均 SSIM（仅作看图锚点，样本少不等于全局） ==")
    for lab in labels[2:]:
        v = per_col[lab]
        if v:
            print(f"   {lab:<10} mean_ssim={sum(v)/len(v):.4f}   per-sample={[round(x,3) for x in v]}")


if __name__ == "__main__":
    main()
