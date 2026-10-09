#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v68_g05_allckpt.py — 在固定推理设置 (CFG=0.7, SDE gamma=0.5) 下评测 v68 的**全部 ckpt**，
得到「SDE 最优设置下 v68 的完整训练轨迹」。

设计要点（为了快）：
  * 模型 / VAE / LPIPS 只加载一次，之后只换 state_dict —— 避免 40 次重建（省 ~15 分钟）；
  * 每点增量落盘 (json + csv)，中断可续（--resume 跳过已完成的 step）；
  * 单点失败不终止整轮。

用法:
  python tools/eval_v68_g05_allckpt.py --cfg 0.7 --gamma 0.5 --steps 50
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import torch

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)

from src.model import DiT_2Cond_models                      # noqa: E402
from src.eval.in_mem_eval import run_in_mem_eval            # noqa: E402

EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"


class ArgsObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=f"{ROOT}/assets/runs/v68_family/v68_aug_sp_c2ot/"
                                         "20261006-192021-v68-aug-sp-c2ot/checkpoints")
    p.add_argument("--config", default=f"{ROOT}/src/train/configs/v68_aug_sp_c2ot.json")
    p.add_argument("--vae-path", default=f"{ROOT}/data/pretrained/pretrained_models/sd-vae-ft-ema")
    p.add_argument("--out-dir", default=f"{ROOT}/experiments/v68_g05_allckpt")
    p.add_argument("--cfg", type=float, default=0.7)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


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


def load_into(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck.get("ema") or ck.get("model") or ck
    if hasattr(st, "state_dict"):
        st = st.state_dict()
    st = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in st.items()}
    m, u = model.load_state_dict(st, strict=False)
    return len(m), len(u)


def dump(rows, out_dir, name):
    if not rows:
        return
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = json.load(open(a.config, encoding="utf-8"))
    cfg.update({
        "eval_csv": EVAL_CSV,
        "in_mem_eval_sets": f"eval200fix:{EVAL_CSV}:187",
        "triple_table_prefix": TABLES + "/",
        "char_remap_json": f"{TABLES}/char_remap.json",
        "callig_remap_json": f"{TABLES}/callig_remap.json",
        "font_remap_json": f"{TABLES}/font_remap.json",
        "vae_path": a.vae_path, "eval_vae_path": a.vae_path,
        "calli_decode_noise": False,
        "eval_cfg": a.cfg, "sde_gamma": a.gamma, "eval_steps": a.steps,
        "in_mem_eval_batch": a.eval_batch, "in_mem_eval_vae_batch": a.eval_batch,
        "in_mem_eval_save_samples": False,
    })
    args_obj_base = ArgsObject(cfg)

    ckpts = []
    for p in glob.glob(os.path.join(a.ckpt_dir, "*.pt")):
        m = re.search(r"(\d+)\.pt$", os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    print("=" * 100)
    print(f"【v68 全 ckpt 轨迹】{len(ckpts)} 个 ckpt | CFG={a.cfg} gamma={a.gamma} steps={a.steps} "
          f"| N=187 | eval-batch={a.eval_batch}")
    print(f"  VAE = {a.vae_path}")
    print("=" * 100, flush=True)

    done = {}
    jp = os.path.join(a.out_dir, "v68_g05_traj.json")
    if a.resume and os.path.exists(jp):
        for r in json.load(open(jp, encoding="utf-8")):
            done[int(r["step"])] = r
        print(f"  [resume] 已有 {len(done)} 个 step", flush=True)

    model = build_model(cfg, device)
    rows = [done[s] for s in sorted(done)]
    n_skip = 0
    t_start = time.time()
    total_todo = sum(1 for s, _ in ckpts if s not in done)
    idx = 0

    for step, path in ckpts:
        if step in done:
            n_skip += 1
            continue
        idx += 1
        t0 = time.time()
        try:
            miss, unexp = load_into(model, path)
            model.eval()
            res = run_in_mem_eval(model, args_obj_base, step=step, device=device,
                                  results_dir=os.path.join(a.out_dir, f"step{step:07d}"), logger=lambda *x: None)
            e = (res or {}).get("eval200fix", {}) or {}
            row = {
                "step": step,
                "ssim": round(float(e.get("ssim", 0) or 0), 4),
                "ssim_med": round(float(e.get("ssim_med", 0) or 0), 4),
                "lpips": round(float(e.get("lpips", 0) or 0), 4),
                "mse": round(float(e.get("mse", 0) or 0), 4),
                "ink_iou": round(float(e.get("ink_iou", 0) or 0), 4),
                "ink_ssim": round(float(e.get("ink_ssim", 0) or 0), 4),
                "skel_iou": round(float(e.get("skel_iou", 0) or 0), 4),
                "frag": round(float(e.get("frag", 0) or 0), 4),
                "n_missing": miss, "n_unexpected": unexp,
                "time_s": round(time.time() - t0, 1),
                "cfg": a.cfg, "gamma": a.gamma, "steps": a.steps,
            }
            rows.append(row)
            rows.sort(key=lambda x: x["step"])
            dump(rows, a.out_dir, "v68_g05_traj")
            el = time.time() - t_start
            eta = (el / idx) * (total_todo - idx) / 60 if idx else 0
            print(f"[{idx:02d}/{total_todo}] step {step:>7} | ssim {row['ssim']:.4f} lpips {row['lpips']:.4f} "
                  f"frag {row['frag']:.3f} ink {row['ink_iou']:.4f} | {row['time_s']}s | ETA {eta:.1f} min", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[{idx:02d}/{total_todo}] step {step:>7} FAILED: {type(ex).__name__}: {ex}", flush=True)

    print("\n" + "=" * 96)
    print("【v68 @ CFG=%.2f gamma=%.2f 全 ckpt 轨迹】" % (a.cfg, a.gamma))
    print(f"{'step':>7} {'ssim':>8} {'lpips':>8} {'mse':>8} {'ink_iou':>8} {'skel':>8} {'frag':>7}")
    for r in rows:
        print(f"{r['step']:>7} {r['ssim']:>8.4f} {r['lpips']:>8.4f} {r['mse']:>8.4f} "
              f"{r['ink_iou']:>8.4f} {r['skel_iou']:>8.4f} {r['frag']:>7.3f}")
    if rows:
        b = max(rows, key=lambda x: x["ssim"])
        print(f"\n  max-SSIM @ step {b['step']}: {b['ssim']:.4f} (lpips {b['lpips']:.4f} frag {b['frag']:.3f})")
    print(f"  total {(time.time()-t_start)/60:.1f} min, skipped {n_skip}, points {len(rows)}")
    print(f"  -> {a.out_dir}/v68_g05_traj.csv")


if __name__ == "__main__":
    main()
