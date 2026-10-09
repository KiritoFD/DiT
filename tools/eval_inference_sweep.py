#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_inference_sweep.py — 通用推理参数扫描器（4090），CFG × SDE gamma (+ 步数)

与 eval_v72_sweep.py 的关系：后者是 v72/Calli-VAE 专用；本脚本把 **ckpt / config / VAE** 全部参数化，
用于在**同一套 harness** 下重测其它模型（例如 v68 + SD-VAE），使跨模型比较真正同口径。

用法示例（v68 复测）:
  python tools/eval_inference_sweep.py \
    --ckpt assets/runs/v68_family/v68_aug_sp_c2ot/<ts>/checkpoints/0200000.pt \
    --config src/train/configs/v68_aug_sp_c2ot.json \
    --vae-path data/pretrained/sd-vae-ft-ema \
    --out-dir experiments/v68_sweep_200k_sameharness --step 200000 --phase 2d
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

EVAL_CSV = f"{ROOT}/exp-std/csv/eval200_fixed.csv"
TABLES = f"{ROOT}/assets/triple_tables_best_minimal"


class ArgsObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--vae-path", default=f"{ROOT}/data/pretrained/sd-vae-ft-ema")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--step", type=int, default=0, help="记录用的训练步")
    p.add_argument("--phase", choices=["2d", "steps", "both"], default="2d")
    p.add_argument("--cfgs", default="0.4,0.6,0.7,0.8,1.0,1.2")
    p.add_argument("--gammas", default="0.0,0.20,0.35,0.50")
    p.add_argument("--steps-list", default="10,20,30,50,100")
    p.add_argument("--steps", type=int, default=50, help="2D 阶段固定步数")
    p.add_argument("--tag-prefix", default="")
    p.add_argument("--eval-batch", type=int, default=16, help="in-mem eval 采样/解码 batch（调大=更快）")
    p.add_argument("--calli-decode-noise", type=int, default=0)
    p.add_argument("--skip-repro", action="store_true")
    p.add_argument("--repro-cfg", type=float, default=1.0)
    p.add_argument("--repro-gamma", type=float, default=0.0)
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
    print(f"  [ckpt] {os.path.basename(ckpt_path)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def dump(rows, out_dir, name):
    if not rows:
        return
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def eval_point(model, base_cfg, device, out_dir, step, cfg_val, gamma_val, n_steps, tag):
    c = copy.deepcopy(base_cfg)
    c["eval_cfg"] = cfg_val
    c["sde_gamma"] = gamma_val
    c["eval_steps"] = n_steps
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    res = run_in_mem_eval(model, ArgsObject(c), step=step, device=device, results_dir=d, logger=print)
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
    raw_cfg["eval_csv"] = EVAL_CSV
    raw_cfg["in_mem_eval_sets"] = f"eval200fix:{EVAL_CSV}:187"
    raw_cfg["triple_table_prefix"] = TABLES + "/"
    raw_cfg["char_remap_json"] = f"{TABLES}/char_remap.json"
    raw_cfg["callig_remap_json"] = f"{TABLES}/callig_remap.json"
    raw_cfg["font_remap_json"] = f"{TABLES}/font_remap.json"
    vae = a.vae_path if os.path.isabs(a.vae_path) else os.path.join(ROOT, a.vae_path)
    raw_cfg["vae_path"] = vae
    raw_cfg["eval_vae_path"] = vae
    raw_cfg["calli_decode_noise"] = bool(a.calli_decode_noise)
    raw_cfg["in_mem_eval_batch"] = a.eval_batch
    raw_cfg["in_mem_eval_vae_batch"] = a.eval_batch
    raw_cfg["in_mem_eval_save_samples"] = False

    print("=" * 100)
    print(f"【通用推理扫描】ckpt={a.ckpt}")
    print(f"  config={a.config}\n  VAE={vae}\n  step={a.step} phase={a.phase} out={a.out_dir}")
    print("=" * 100, flush=True)

    model = load_ema(build_model(raw_cfg, device), a.ckpt)
    pre = a.tag_prefix

    if not a.skip_repro:
        print(f"\n[Phase 0] 复现点 CFG={a.repro_cfg} gamma={a.repro_gamma} steps=50", flush=True)
        chk = eval_point(model, raw_cfg, device, a.out_dir, a.step, a.repro_cfg, a.repro_gamma, 50,
                         f"{pre}check_cfg{a.repro_cfg:.2f}_gamma{a.repro_gamma:.2f}")
        print(f"  >>> ssim={chk['ssim']:.4f} lpips={chk['lpips']:.4f} frag={chk['frag']:.3f}", flush=True)
        dump([chk], a.out_dir, f"{pre}repro_check")

    if a.phase in ("2d", "both"):
        cfgs = [float(x) for x in a.cfgs.split(",") if x.strip()]
        gammas = [float(x) for x in a.gammas.split(",") if x.strip()]
        rows, i, total = [], 0, len(cfgs) * len(gammas)
        for c in cfgs:
            for g in gammas:
                i += 1
                tag = f"{pre}cfg{c:.2f}_gamma{g:.2f}"
                print(f"\n>>> [{i:02d}/{total}] {tag} ...", flush=True)
                r = eval_point(model, raw_cfg, device, a.out_dir, a.step, c, g, a.steps, tag)
                rows.append(r)
                dump(rows, a.out_dir, f"{pre}sweep_2d")
                print(f"  ok {tag}: ssim={r['ssim']:.4f} lpips={r['lpips']:.4f} frag={r['frag']:.3f} ({r['time_s']}s)", flush=True)
        print("\n" + "=" * 96)
        print("【Phase 1 总表】")
        print(f"{'cfg':>5} {'gamma':>6} | {'ssim':>8} {'lpips':>8} {'mse':>8} {'ink_iou':>8} {'frag':>7}")
        for r in rows:
            print(f"{r['cfg']:>5.2f} {r['gamma']:>6.2f} | {r['ssim']:>8.4f} {r['lpips']:>8.4f} "
                  f"{r['mse']:>8.4f} {r['ink_iou']:>8.4f} {r['frag']:>7.3f}")
        bs = max(rows, key=lambda x: x["ssim"]); bl = min(rows, key=lambda x: x["lpips"])
        print(f"\n  max-SSIM : cfg={bs['cfg']:.2f} gamma={bs['gamma']:.2f} -> {bs['ssim']:.4f} (lpips {bs['lpips']:.4f})")
        print(f"  min-LPIPS: cfg={bl['cfg']:.2f} gamma={bl['gamma']:.2f} -> {bl['lpips']:.4f} (ssim {bl['ssim']:.4f})")
        json.dump({"best_ssim": bs, "best_lpips": bl},
                  open(os.path.join(a.out_dir, f"{pre}sweep_2d_best.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    if a.phase in ("steps", "both"):
        bc, bg = 0.7, 0.5
        jp = os.path.join(a.out_dir, f"{pre}sweep_2d.json")
        if os.path.exists(jp):
            b = max(json.load(open(jp, encoding="utf-8")), key=lambda x: x["ssim"])
            bc, bg = b["cfg"], b["gamma"]
        print(f"\n[Phase 2] 步数扫描 @ cfg={bc:.2f} gamma={bg:.2f}", flush=True)
        srows = []
        for s in [int(x) for x in a.steps_list.split(",") if x.strip()]:
            r = eval_point(model, raw_cfg, device, a.out_dir, a.step, bc, bg, s, f"{pre}steps{s}")
            srows.append(r); dump(srows, a.out_dir, f"{pre}sweep_steps")
            print(f"  ok steps={s}: ssim={r['ssim']:.4f} lpips={r['lpips']:.4f} ({r['time_s']}s)", flush=True)
        print(f"\n【Phase 2 总表】(cfg={bc:.2f} gamma={bg:.2f})")
        for r in srows:
            print(f"  steps={r['steps']:>4} nfe={r['steps']*2:>4} ssim={r['ssim']:.4f} lpips={r['lpips']:.4f} frag={r['frag']:.3f}")

    print(f"\n[ALL DONE] {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
