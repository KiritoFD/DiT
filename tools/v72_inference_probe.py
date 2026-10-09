#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v72_inference_probe.py — v72 的推理端对症实验（A/B/C 一次跑完）

针对已诊断出的欠缺（**高频细节只有真迹 0.67×、墨量 0.75×**），扫三条纯推理轴：

  A. 解码端后验再注噪   calli_decode_noise ∈ {0, 1}
       理由: 训练目标 = posterior.sample()*sf（含 σ≈0.39 噪声），推理却把均值喂 decoder
  B. 积分步数           steps ∈ {10,20,30,50,100,200}
       理由: Heun-50 的截断误差本身就是低通；若步数能补高频 -> 换求解器即免费午餐
  C. 潜空间尺度标定     latent_scale ∈ {1.0, 1.15, 1.30}
       理由: 若预测 latent 幅度系统性偏小(回归到均值)，乘 ρ 后解码可补墨量/对比度

每个配置都**存图**，再对图直接量 ink / 高频 / 梯度 / Laplacian —— 不只信 SSIM。
基线：cfg=0.7, gamma=0.0（v72 的 ODE 最优点附近），N=40（eval200_fixed 均布抽样）。
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"
CALLI_VAE = f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
V72_CKPT = (f"{ROOT}/assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/"
            "20261009-142151-v72-callivae-kl1e6-s12500-frozen-sp-c2ot/checkpoints/0130000.pt")
V72_CFG = f"{ROOT}/src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json"


class ArgsObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


# ---------------- 图像统计（与 census 工具同口径） ----------------
def img_stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ink = float((a < 128).mean())
    b = a / 255.0
    gx = np.zeros_like(b); gy = np.zeros_like(b)
    gx[:, 1:-1] = b[:, 2:] - b[:, :-2]
    gy[1:-1, :] = b[2:, :] - b[:-2, :]
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    lap = np.zeros_like(b)
    lap[1:-1, 1:-1] = (4 * b[1:-1, 1:-1] - b[:-2, 1:-1] - b[2:, 1:-1]
                       - b[1:-1, :-2] - b[1:-1, 2:])
    lapvar = float(lap.var())
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    hf = float(F[r > 0.5].sum() / max(F.sum(), 1e-9))
    return ink, grad, lapvar, hf, float(b.mean()), float(b.std())


def build_model(cfg, device):
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


def pick_csv(n):
    rows = list(csv.DictReader(open(EVAL_CSV, encoding="utf-8")))
    total = len(rows)
    idx = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    out = f"{ROOT}/exp-std/csv/probe{n}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows([rows[i] for i in idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--out", default=f"{ROOT}/experiments/v72_probe")
    ap.add_argument("--axes", default="ABC", help="要跑哪些轴: A(解码注噪) B(步数) C(潜空间尺度)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda:0")

    csvp = pick_csv(a.n)
    sets = f"probe:{csvp}:{a.n}"
    cfg = json.load(open(V72_CFG, encoding="utf-8"))
    cfg.update({
        "eval_csv": EVAL_CSV, "in_mem_eval_sets": sets,
        "triple_table_prefix": TABLES + "/",
        "char_remap_json": f"{TABLES}/char_remap.json",
        "callig_remap_json": f"{TABLES}/callig_remap.json",
        "font_remap_json": f"{TABLES}/font_remap.json",
        "vae_path": CALLI_VAE, "eval_vae_path": CALLI_VAE,
        "eval_cfg": a.cfg, "sde_gamma": a.gamma,
        "in_mem_eval_batch": 16, "in_mem_eval_vae_batch": 16,
        "in_mem_eval_save_samples": True,
    })

    # ---- 预加载并【包装】VAE decode，以便施加 latent_scale（monkey-patch，不改源码） ----
    from src.eval.inference import load_eval_vae
    vae = load_eval_vae(device, CALLI_VAE)
    holder = {"s": 1.0}
    orig_decode = vae.decode

    def patched_decode(z, *args, **kw):
        if holder["s"] != 1.0:
            z = z * holder["s"]
        return orig_decode(z, *args, **kw)

    vae.decode = patched_decode
    print(f"[vae] patched decode, base scale = 1.0  ({CALLI_VAE})", flush=True)

    model = build_model(cfg, device)
    ck = torch.load(V72_CKPT, map_location="cpu", weights_only=False)
    st = ck.get("ema") or ck.get("model") or ck
    if hasattr(st, "state_dict"):
        st = st.state_dict()
    st = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in st.items()}
    m, u = model.load_state_dict(st, strict=False)
    print(f"[ckpt] {os.path.basename(V72_CKPT)} missing={m} unexpected={u}", flush=True)
    model.eval()

    configs = []
    if "B" in a.axes:
        for s in (10, 20, 30, 50, 100, 200):        # B: 步数
            configs.append(dict(tag=f"B_steps{s}", steps=s, noise=0, scale=1.0))
    if "A" in a.axes:
        configs.append(dict(tag="A_noise1", steps=50, noise=1, scale=1.0))   # A: 解码端后验再注噪
    if "C" in a.axes:
        for sc in (1.15, 1.30):                                              # C: 潜空间尺度
            configs.append(dict(tag=f"C_scale{sc}", steps=50, noise=0, scale=sc))

    set_name = sets.split(":")[0]                   # ★ 结果 dict 以【评测集名】为键
    rows = []
    gt_buf = {"m": None}                            # GT 统计只算一次（各配置的 GT 相同）
    for i, conf in enumerate(configs, 1):
        holder["s"] = conf["scale"]
        c = copy.deepcopy(cfg)
        c["eval_steps"] = conf["steps"]
        c["calli_decode_noise"] = bool(conf["noise"])
        d = os.path.join(a.out, conf["tag"])
        t0 = time.time()
        res = run_in_mem_eval(model, ArgsObject(c), step=130000, device=device,
                              results_dir=d, logger=lambda *x: None)
        e = (res or {}).get(set_name, {}) or {}          # ★ 用 set_name
        sd = os.path.join(d, "eval_samples_ctrl", "step0130000", set_name)
        out_st, gt_here = [], []
        for k in range(a.n):
            op = os.path.join(sd, f"g{k}.png")
            gp = os.path.join(sd, f"gt{k}.png")
            if os.path.exists(op):
                out_st.append(img_stats(op))
            if os.path.exists(gp):
                gt_here.append(img_stats(gp))
        if gt_buf["m"] is None and gt_here:
            gt_buf["m"] = np.array(gt_here).mean(axis=0)
        gt_m = gt_buf["m"] if gt_buf["m"] is not None else np.zeros(6)
        o = np.array(out_st).mean(axis=0) if out_st else np.zeros(6)
        row = {
            "tag": conf["tag"], "steps": conf["steps"], "noise": conf["noise"], "scale": conf["scale"],
            "ssim": round(float(e.get("ssim", 0) or 0), 4),
            "lpips": round(float(e.get("lpips", 0) or 0), 4),
            "ink_iou": round(float(e.get("ink_iou", 0) or 0), 4),
            "frag": round(float(e.get("frag", 0) or 0), 4),
            "mse": round(float(e.get("mse", 0) or 0), 4),
            "img_ink": round(float(o[0]), 4),
            "img_ink_ratio": round(float(o[0] / gt_m[0]), 3) if gt_m[0] else 0,
            "img_grad": round(float(o[1]), 4),
            "img_lapvar": round(float(o[2]), 6),
            "img_hf": round(float(o[3]), 4),
            "img_hf_ratio": round(float(o[3] / gt_m[3]), 3) if gt_m[3] else 0,
            "img_std": round(float(o[5]), 2),
            "t_s": round(time.time() - t0, 1),
        }
        rows.append(row)
        with open(os.path.join(a.out, "probe_results.json"), "w", encoding="utf-8") as f:
            json.dump({"gt": {"ink": round(float(gt_m[0]), 4), "grad": round(float(gt_m[1]), 4),
                              "lapvar": round(float(gt_m[2]), 6), "hf": round(float(gt_m[3]), 4),
                              "std": round(float(gt_m[5]), 2)}, "rows": rows}, f,
                      indent=2, ensure_ascii=False)
        print(f"[{i:02d}/{len(configs)}] {conf['tag']:<12} ssim={row['ssim']:.4f} lpips={row['lpips']:.4f} "
              f"ink={row['img_ink_ratio']:.2f}x hf={row['img_hf_ratio']:.2f}x frag={row['frag']:.3f} ({row['t_s']}s)",
              flush=True)

    print("\n== GT 基准（绝对量，用于核对） ==")
    print(f"   ink={gt_m[0]:.4f} grad={gt_m[1]:.4f} lapvar={gt_m[2]:.6f} hf={gt_m[3]:.4f} mean={gt_m[4]:.1f} std={gt_m[5]:.2f}")
    print(f"\n{'tag':<14}{'steps':>6}{'noise':>6}{'scale':>6} | {'ssim':>8}{'lpips':>8}{'ink_iou':>8}{'frag':>7} | "
          f"{'gen_ink':>8}{'ink×GT':>7}{'gen_hf':>8}{'hf×GT':>7}{'std':>7}")
    for r in rows:
        print(f"{r['tag']:<14}{r['steps']:>6}{r['noise']:>6}{r['scale']:>6} | {r['ssim']:>8.4f}{r['lpips']:>8.4f}"
              f"{r['ink_iou']:>8.4f}{r['frag']:>7.3f} | {r['img_ink']:>8.4f}{r['img_ink_ratio']:>7.2f}"
              f"{r['img_hf']:>8.4f}{r['img_hf_ratio']:>7.2f}{r['img_std']:>7.1f}")
    print(f"\n-> {a.out}/probe_results.json")


if __name__ == "__main__":
    main()
