#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v72_sweep.py — v72 (冻结 Calli-VAE 潜空间) 的推理参数扫描（4090 版）

与 tools/eval_2d_sweep_cfg_sde.py 的区别（后者是为 48 + SD-VAE 写死的）：
  * 路径全部指向 4090 的 /root/Workspace/xy/DiT
  * VAE 指向**冻结的新 Calli-VAE**（calli_vae_kl1e6_s12500_stdconv，标准约定 decode(z/sf)）
  * 内置 **复现校验点**：CFG=1.0 / γ=0.0 应与训练内评测 (0.6070 @130k) 一致
  * 两阶段：Phase 1 = CFG × γ 二维；Phase 2 = 在最优 (CFG,γ) 上扫步数

用法（在 4090）:
  python tools/eval_v72_sweep.py --phase 2d
  python tools/eval_v72_sweep.py --phase steps
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import torch

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

VAE = f"{ROOT}/experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv"
EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"


class ArgsObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=f"{ROOT}/assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/"
                                     "20261009-142151-v72-callivae-kl1e6-s12500-frozen-sp-c2ot/checkpoints/0130000.pt")
    p.add_argument("--config", default=f"{ROOT}/src/train/configs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot.json")
    p.add_argument("--out-dir", default=f"{ROOT}/experiments/v72_sweep_130k")
    p.add_argument("--step", type=int, default=130000, help="用于结果目录/记录的训练步")
    p.add_argument("--phase", choices=["2d", "steps", "both"], default="2d")
    p.add_argument("--cfgs", default="0.4,0.6,0.7,0.8,1.0,1.2")
    p.add_argument("--gammas", default="0.0,0.20,0.35,0.50")
    p.add_argument("--steps-list", default="10,20,30,50,100")
    p.add_argument("--steps", type=int, default=50, help="2D 阶段固定步数")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_model(raw_cfg, device):
    model = DiT_2Cond_models[raw_cfg.get("model", "DiT-2Cond-Sp/2")](
        learn_sigma=False,
        norm_type=raw_cfg.get("norm_type", "layer"),
        mlp_type=raw_cfg.get("mlp_type", "gelu"),
        qk_norm=raw_cfg.get("qk_norm", 0),
        rope=raw_cfg.get("rope", 0),
        condition_fusion=raw_cfg.get("condition_fusion", "factorized_cat"),
        cond_fusion_norm=raw_cfg.get("cond_fusion_norm", "split"),
        num_calligraphers=raw_cfg.get("num_calligraphers", 10),
        num_characters=raw_cfg.get("num_characters", 4690),
        num_script_classes=raw_cfg.get("num_script_classes", 3),
        use_script_cond=raw_cfg.get("use_script_cond", True),
        char_embed_dim=raw_cfg.get("char_embed_dim", 256),
        callig_embed_dim=raw_cfg.get("callig_embed_dim", 32),
        script_embed_dim=raw_cfg.get("script_embed_dim", 16),
        glyph_inject_layers=raw_cfg.get("glyph_inject_layers", 0),
        cond_inject_at=raw_cfg.get("cond_inject_at", "2,4,5,6"),
        cond_inject_scale=raw_cfg.get("cond_inject_scale", True),
    ).to(device)
    return model


def load_ema(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    clean = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"  [ckpt] loaded {os.path.basename(ckpt_path)}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def dump(rows, out_dir, name):
    if not rows:
        return
    jp = os.path.join(out_dir, f"{name}.json")
    cp = os.path.join(out_dir, f"{name}.csv")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    keys = list(rows[0].keys())
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def eval_point(model, base_cfg, device, out_dir, step, cfg_val, gamma_val, n_steps, tag):
    cfg_copy = copy.deepcopy(base_cfg)
    cfg_copy["eval_cfg"] = cfg_val
    cfg_copy["sde_gamma"] = gamma_val
    cfg_copy["eval_steps"] = n_steps
    pt_dir = os.path.join(out_dir, tag)
    os.makedirs(pt_dir, exist_ok=True)
    t0 = time.time()
    res = run_in_mem_eval(model, ArgsObject(cfg_copy), step=step, device=device,
                          results_dir=pt_dir, logger=print)
    dt = time.time() - t0
    e = (res or {}).get("eval200fix", {}) or {}
    return {
        "cfg": cfg_val, "gamma": gamma_val, "steps": n_steps, "tag": tag,
        "ssim": round(float(e.get("ssim", 0) or 0), 4),
        "ssim_med": round(float(e.get("ssim_med", 0) or 0), 4),
        "lpips": round(float(e.get("lpips", 0) or 0), 4),
        "mse": round(float(e.get("mse", 0) or 0), 4),
        "ink_iou": round(float(e.get("ink_iou", 0) or 0), 4),
        "ink_ssim": round(float(e.get("ink_ssim", 0) or 0), 4),
        "skel_iou": round(float(e.get("skel_iou", 0) or 0), 4),
        "frag": round(float(e.get("frag", 0) or 0), 4),
        "time_s": round(dt, 1),
    }


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    raw_cfg = json.load(open(a.config, encoding="utf-8"))
    # --- 4090 + Calli-VAE 路径纠正（不改动原配置语义，只改路径与评测开关）---
    raw_cfg["eval_csv"] = EVAL_CSV
    raw_cfg["in_mem_eval_sets"] = f"eval200fix:{EVAL_CSV}:187"
    raw_cfg["triple_table_prefix"] = TABLES + "/"
    raw_cfg["char_remap_json"] = f"{TABLES}/char_remap.json"
    raw_cfg["callig_remap_json"] = f"{TABLES}/callig_remap.json"
    raw_cfg["font_remap_json"] = f"{TABLES}/font_remap.json"
    raw_cfg["vae_path"] = VAE
    raw_cfg["eval_vae_path"] = VAE
    raw_cfg["calli_decode_noise"] = False           # 新 Calli-VAE 是标准约定，无需注噪补丁
    raw_cfg["in_mem_eval_batch"] = 16
    raw_cfg["in_mem_eval_vae_batch"] = 16
    raw_cfg["in_mem_eval_save_samples"] = False

    print("=" * 100)
    print("【v72 推理参数扫描】ckpt =", a.ckpt)
    print(f"  VAE   = {VAE}")
    print(f"  eval  = {EVAL_CSV} (N=187)   phase={a.phase}   out={a.out_dir}")
    print("=" * 100, flush=True)

    model = build_model(raw_cfg, device)
    load_ema(model, a.ckpt)

    # ---------- Phase 0: 复现校验（必须与训练内评测一致） ----------
    print("\n[Phase 0] 复现校验点 CFG=1.0 / gamma=0.0 / steps=50  (训练内评测 @130k = 0.6070)", flush=True)
    chk = eval_point(model, raw_cfg, device, a.out_dir, a.step, 1.0, 0.0, 50, "check_cfg1.00_gamma0.00")
    print(f"  >>> 复现: ssim={chk['ssim']:.4f}  (期望 ~0.6070)  lpips={chk['lpips']:.4f}  frag={chk['frag']:.3f}", flush=True)
    dump([chk], a.out_dir, "repro_check")

    if a.phase in ("2d", "both"):
        cfgs = [float(x) for x in a.cfgs.split(",") if x.strip()]
        gammas = [float(x) for x in a.gammas.split(",") if x.strip()]
        rows, i, total = [], 0, len(cfgs) * len(gammas)
        print(f"\n[Phase 1] CFG x gamma 网格 {len(cfgs)}x{len(gammas)}={total} 点, steps={a.steps}", flush=True)
        for c in cfgs:
            for g in gammas:
                i += 1
                tag = f"cfg{c:.2f}_gamma{g:.2f}"
                if c == 1.0 and g == 0.0:
                    rows.append(chk); print(f"  [{i:02d}/{total}] {tag} 复用 Phase0 结果 -> ssim={chk['ssim']:.4f}", flush=True)
                    dump(rows, a.out_dir, "sweep_2d"); continue
                print(f"\n>>> [{i:02d}/{total}] {tag} (N=187) ...", flush=True)
                r = eval_point(model, raw_cfg, device, a.out_dir, a.step, c, g, a.steps, tag)
                rows.append(r)
                dump(rows, a.out_dir, "sweep_2d")
                print(f"  ok {tag}: ssim={r['ssim']:.4f} lpips={r['lpips']:.4f} frag={r['frag']:.3f} ({r['time_s']}s)", flush=True)
        print("\n" + "=" * 96)
        print("【Phase 1 总表】")
        print(f"{'cfg':>5} {'gamma':>6} | {'ssim':>8} {'ssim_med':>9} {'lpips':>8} {'mse':>8} {'ink_iou':>8} {'frag':>7} {'skel':>8}")
        for r in rows:
            print(f"{r['cfg']:>5.2f} {r['gamma']:>6.2f} | {r['ssim']:>8.4f} {r['ssim_med']:>9.4f} {r['lpips']:>8.4f} "
                  f"{r['mse']:>8.4f} {r['ink_iou']:>8.4f} {r['frag']:>7.3f} {r['skel_iou']:>8.4f}")
        best_s = max(rows, key=lambda x: x["ssim"]); best_l = min(rows, key=lambda x: x["lpips"])
        print(f"\n  max-SSIM  : cfg={best_s['cfg']:.2f} gamma={best_s['gamma']:.2f} -> {best_s['ssim']:.4f} (lpips {best_s['lpips']:.4f})")
        print(f"  min-LPIPS : cfg={best_l['cfg']:.2f} gamma={best_l['gamma']:.2f} -> {best_l['lpips']:.4f} (ssim {best_l['ssim']:.4f})")
        print("=" * 96, flush=True)
        json.dump({"best_ssim": best_s, "best_lpips": best_l},
                  open(os.path.join(a.out_dir, "sweep_2d_best.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if a.phase in ("steps", "both"):
        # 最优 CFG/gamma：优先取 2D 结果，否则用手动默认
        bc, bg = 0.8, 0.5
        jp = os.path.join(a.out_dir, "sweep_2d.json")
        if os.path.exists(jp):
            r2 = json.load(open(jp, encoding="utf-8"))
            b = max(r2, key=lambda x: x["ssim"])
            bc, bg = b["cfg"], b["gamma"]
        print(f"\n[Phase 2] 步数扫描 @ cfg={bc:.2f} gamma={bg:.2f}", flush=True)
        srows = []
        for s in [int(x) for x in a.steps_list.split(",") if x.strip()]:
            tag = f"steps{s}"
            print(f"\n>>> {tag} ...", flush=True)
            r = eval_point(model, raw_cfg, device, a.out_dir, a.step, bc, bg, s, tag)
            srows.append(r); dump(srows, a.out_dir, "sweep_steps")
            print(f"  ok steps={s}: ssim={r['ssim']:.4f} lpips={r['lpips']:.4f} ({r['time_s']}s)", flush=True)
        print("\n【Phase 2 总表】(cfg=%.2f gamma=%.2f)" % (bc, bg))
        print(f"{'steps':>6} {'nfe':>6} | {'ssim':>8} {'lpips':>8} {'frag':>7}")
        for r in srows:
            print(f"{r['steps']:>6} {r['steps']*2:>6} | {r['ssim']:>8.4f} {r['lpips']:>8.4f} {r['frag']:>7.3f}")

    print(f"\n[ALL DONE] {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
