#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""latent_hf_probe.py — 探针: DiT 预测出的 latent 本身缺不缺高频？

做法: 在评测链路里**包一层 vae.decode**, 截获 DiT 送进解码器的 latent(= 预测的 z_stored),
      再与"同一批 GT 图编码得到的 latent"比较顶带能量份额。

判读:
  * 若 pred_latent 的顶带份额 << GT latent 的 -> **DiT 预测的 latent 本身就低频**,
    说明绑定约束在 DiT 的目标函数(而非 VAE 解码能力), 且"图像空间/目标函数"才是该下手的地方;
  * 若两者接近 -> DiT 的 latent 高频没问题, 缺口出在解码器对该 latent 的还原上。

口径: 与离线普查一致 —— 灰度(这里=通道均值)去均值后 FFT, r>0.5 (按 (H/2) 归一, 不再除 r.max())。
      份额是尺度无关量, 故 latent 的 sf 缩放不影响结论。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vae"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402
from hf_losses import band_fraction                         # noqa: E402

EVAL_CSV = "exp-std/csv/eval200_fixed.csv"
TABLES = "assets/triple_tables_best_minimal"
SF = 0.18215


class A:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--n", type=int, default=187)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))[:a.n]
    sub = os.path.join(a.out, f"probe{a.n}.csv")
    with open(sub, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- 1. GT latent（同一批图, 同一个 VAE） ----
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae_path).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    gts = [np.asarray(Image.open(r["image_path"]).convert("RGB").resize((256, 256), Image.BICUBIC)) for r in rows]
    zs_sample, zs_mode = [], []
    for i in range(0, len(gts), 8):
        xb = torch.from_numpy(np.stack(gts[i:i + 8])).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(xb).latent_dist
            zs_sample.append(post.sample().float().cpu())
            zs_mode.append(post.mode().float().cpu())
    z_s = torch.cat(zs_sample); z_m = torch.cat(zs_mode)
    bf_gt_sample = band_fraction(z_s).mean().item()
    bf_gt_mode = band_fraction(z_m).mean().item()
    print(f"[GT latent] n={z_s.shape[0]}  shape={tuple(z_s.shape[1:])}")
    print(f"  sample: 顶带份额={bf_gt_sample:.4f}  std={z_s.std():.4f}")
    print(f"  mode  : 顶带份额={bf_gt_mode:.4f}  std={z_m.std():.4f}")

    # ---- 2. 截获 DiT 预测的 latent ----
    CAP = []
    orig_decode = vae.decode

    def patched(z, *args, **kw):
        if z.shape[1] == z_s.shape[1]:      # 只截图像通道, 排除 aux/骨架解码
            CAP.append(z.detach().float().cpu())
        return orig_decode(z, *args, **kw)

    vae.decode = patched
    del vae
    torch.cuda.empty_cache()

    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg.update({
        "eval_csv": EVAL_CSV, "in_mem_eval_sets": f"probe:{sub}:{a.n}",
        "triple_table_prefix": TABLES + "/",
        "char_remap_json": f"{TABLES}/char_remap.json",
        "callig_remap_json": f"{TABLES}/callig_remap.json",
        "font_remap_json": f"{TABLES}/font_remap.json",
        "vae_path": a.vae_path, "eval_vae_path": a.vae_path,
        "eval_cfg": a.cfg, "sde_gamma": a.gamma, "eval_steps": a.steps,
        "in_mem_eval_batch": a.batch, "in_mem_eval_vae_batch": a.batch,
        "in_mem_eval_save_samples": False,
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
    print(f"[ckpt] {os.path.basename(a.ckpt)} missing={m} unexpected={u}")
    model.eval()

    run_in_mem_eval(model, A(cfg), step=0, device=dev, results_dir=a.out, logger=lambda *x: None)
    z_pred = torch.cat(CAP)
    print(f"\n[DiT 预测 latent] 截获 {z_pred.shape[0]} 个  shape={tuple(z_pred.shape[1:])}")

    bf_pred = band_fraction(z_pred).mean().item()
    # 逐通道顶带份额
    per_ch_gt = [band_fraction(z_s[:, c:c+1]).mean().item() for c in range(z_s.shape[1])]
    per_ch_pr = [band_fraction(z_pred[:, c:c+1]).mean().item() for c in range(z_pred.shape[1])]

    print(f"\n{'':<16}{'顶带份额':>10}{'std':>9}{'mean':>9}   相对 GT latent")
    print(f"{'GT latent(sample)':<16}{bf_gt_sample:>10.4f}{z_s.std():>9.4f}{z_s.mean():>9.4f}{1.0:>13.3f}")
    print(f"{'GT latent(mode)':<16}{bf_gt_mode:>10.4f}{z_m.std():>9.4f}{z_m.mean():>9.4f}"
          f"{bf_gt_mode/bf_gt_sample:>13.3f}")
    print(f"{'DiT 预测':<16}{bf_pred:>10.4f}{z_pred.std():>9.4f}{z_pred.mean():>9.4f}"
          f"{bf_pred/bf_gt_sample:>13.3f}")
    print(f"\n逐通道顶带份额:")
    print(f"  GT  : {[round(x,4) for x in per_ch_gt]}")
    print(f"  pred: {[round(x,4) for x in per_ch_pr]}")
    res = {"ckpt": a.ckpt, "n": int(z_pred.shape[0]),
           "gt_sample_bandfrac": bf_gt_sample, "gt_mode_bandfrac": bf_gt_mode,
           "pred_bandfrac": bf_pred, "pred_over_gt": bf_pred / bf_gt_sample,
           "gt_std": float(z_s.std()), "pred_std": float(z_pred.std()),
           "gt_mean": float(z_s.mean()), "pred_mean": float(z_pred.mean()),
           "per_ch_gt": per_ch_gt, "per_ch_pred": per_ch_pr}
    json.dump(res, open(os.path.join(a.out, "probe.json"), "w", encoding="utf-8"), indent=2)
    print(f"\n-> {a.out}/probe.json")
    print("\n判读: pred_over_gt << 1  => DiT 预测的 latent 本身低频(约束在 DiT 目标);" )
    print("      pred_over_gt ≈ 1  => latent 高频没问题, 缺口在解码/其他环节。")


if __name__ == "__main__":
    main()
