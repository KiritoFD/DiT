# -*- coding: utf-8 -*-
"""
eval_unified.py — 历届预训练实验统一 eval → 汇总 CSV.

同一 eval 集 (eval_strict_midclean), 同一噪声 (seed 0), 同一步数 (50),
每模型用原生采样器 (flow=Euler, ddpm=DDIM), cfg ∈ {1.7, 4.0} 两档.
指标: mse(×4口径)/ssim/lpips/skel_iou + 分书体 ssim.

用法 (远程 GPU):
  /opt/conda/bin/python tools/eval_unified.py \
      --eval-csv 5script/eval_strict_midclean.csv \
      --out-csv 5script/eval_unified_20260829.csv
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from src.model.controlnet import load_main_model
from src.eval.inference import (
    build_diffusion, sample_latents, decode_and_save, compute_metrics,
    load_eval_vae, _ssim,
)

# (实验名, run 目录, ckpt 步数) — best/last ckpt
EXPERIMENTS = [
    ("s19-midclean-flow", "s19_midclean_s_flow/20260828-143711-s19-midclean-s-flow", 50000),
    ("s18-top6-flow", "s18_s_flow_small/20260827-232003-s18-s-flow-small", 43000),
    ("s17-3top30-flow", "s17_s_flow/20260827-070924-s17-s-flow", 165000),
    ("s15-3top30-ws-flow", "s15_ws_flow/20260826-133102-s15-ws-flow", 200000),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", default="5script/eval_strict_midclean.csv")
    ap.add_argument("--out-csv", default="5script/eval_unified_20260829.csv")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[17, 40],
                    help="cfg×10 整数列表, 17=1.7 40=4.0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--vae-batch", type=int, default=32)
    ap.add_argument("--workdir", default="/tmp/eval_unified")
    return ap.parse_args()


def eval_one(run_dir, step, cfgv, args, vae, rows_out):
    cfg = json.load(open(os.path.join(ROOT, "5script/results", run_dir,
                                      "resolved_config.json"), encoding="utf-8"))
    ckpt_path = os.path.join(ROOT, "5script/results", run_dir,
                             "checkpoints", f"{step:07d}.pt")
    device = torch.device("cuda")
    model = load_main_model(
        cfg["model"], ckpt_path, device=device,
        num_calligraphers=cfg.get("num_calligraphers", 1011),
        num_characters=cfg.get("num_characters", 35130),
        condition_fusion=cfg.get("condition_fusion", "factorized_add"),
        callig_embed_dim=cfg.get("callig_embed_dim", 128),
        char_embed_dim=cfg.get("char_embed_dim", 384),
        char_proj_mode=cfg.get("char_proj_mode", "ln_only"),
        freeze_char_table=cfg.get("freeze_table", cfg.get("freeze_char_table", True)),
        use_checkpoint=False, learn_sigma=True)
    model.eval()
    diffusion = build_diffusion(args.steps, cfg.get("diffusion_type", "flow"))

    rows = list(csv.DictReader(open(args.eval_csv, encoding="utf-8")))
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    img_root = cfg.get("img_root", "final_imgs_256")
    gts = torch.stack([
        tf(Image.open(os.path.join(ROOT, img_root,
                                   os.path.basename(r["image_path"]))).convert("RGB"))
        for r in rows])
    conds = [(int(r["calligrapher_id"]), int(r["character_id"])) for r in rows]
    scripts = [r["script"] for r in rows]
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(len(rows), 4, 32, 32, generator=g)

    dec_dir = os.path.join(args.workdir, f"{os.path.basename(run_dir)}_{step}",
                           f"cfg{cfgv}")
    latents = sample_latents(model, diffusion, noise, conds, cfgv,
                             args.batch, device, seed=0)
    n = decode_and_save(vae, latents, 0.18215, dec_dir, "sample",
                        gts=gts, vae_batch=args.vae_batch)
    res = compute_metrics(dec_dir, dec_dir, "sample", n, use_lpips=True)

    per_script = {}
    for i, s in enumerate(scripts):
        p = os.path.join(dec_dir, f"sample{i}.png")
        gt = os.path.join(dec_dir, f"gt{i}.png")
        if not (os.path.exists(p) and os.path.exists(gt)):
            continue
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        b = np.asarray(Image.open(gt).convert("RGB"), np.float32) / 255.0
        per_script.setdefault(s, []).append(_ssim(a, b))

    row = {
        "experiment": cfg.get("experiment_name", run_dir),
        "run_dir": run_dir, "ckpt": step,
        "model": cfg["model"], "diffusion": cfg.get("diffusion_type", "flow"),
        "data": os.path.basename(cfg.get("data_csv", "")),
        "n": n, "cfg": cfgv, "steps": args.steps,
        "mse_x4": round(res.get("mse_mean", float("nan")), 4),
        "ssim": round(res.get("ssim_mean", float("nan")), 4),
        "lpips": round(res.get("lpips_mean", float("nan")), 4),
        "skel_iou": round(res.get("skel_iou_mean", float("nan")), 4),
    }
    for s, v in sorted(per_script.items()):
        row[f"ssim_{s}"] = round(float(np.mean(v)), 4)
    rows_out.append(row)
    print("[result] " + json.dumps(row, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    device = torch.device("cuda")
    vae = load_eval_vae(device, os.path.join(ROOT, "pretrained_models/sd-vae-ft-ema"))
    rows_out = []
    for name, run_dir, step in EXPERIMENTS:
        for cfgv10 in args.cfgs:
            cfgv = cfgv10 / 10.0
            print(f"\n=== {name} ckpt={step} cfg={cfgv} ===", flush=True)
            try:
                eval_one(run_dir, step, cfgv, args, vae, rows_out)
            except Exception as e:
                print(f"[FAILED] {name} cfg={cfgv}: {e}", flush=True)
            if rows_out:
                with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                    w.writeheader()
                    w.writerows(rows_out)
    print(f"\nDone -> {args.out_csv}")


if __name__ == "__main__":
    main()
