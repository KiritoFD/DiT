#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""大测试集本地 GPU 评测: 生成 + 存图 + MSE/SSIM。

复用 auto_eval_cpu 的 build_model/load_vae 与 eval_auto.eval_gen_in_memory,
save_samples_dir 打开后每张生成图与 GT 都会落盘 (out/step<tag>/sample{i}.png, gt{i}.png)。

用法:
  python tools/eval_large.py --ckpt docs/s6_report/ckpt_diffonly_0195000.pt \
      --csv 5script/eval500_top6.csv --tag diffonly_s195000 \
      --out-dir docs/s6_report/large_eval --device cuda --batch 10
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src/ 优先: 根目录的 eval_auto/dataset/models 等是旧拷贝,
# src/ 下才是与远程运行版一致的新版(支持 save_samples_dir 等 kwargs)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from auto_eval_cpu import build_model, load_vae, load_ckpt_weights, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="docs/s6_report/large_eval")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n", type=int, default=None, help="默认用满 csv 全部行")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=10)
    args_cli = ap.parse_args()

    os.chdir(ROOT)
    log(f"[large-eval] ckpt={args_cli.ckpt} csv={args_cli.csv} device={args_cli.device}")

    ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    # 覆盖评测数据源; 关闭 show/seen 展示缓存; 解除 ckpt 存的 eval_n=100 截断
    args.eval_csv = args_cli.csv
    args.show5_csv = None
    args.eval_n = 10 ** 9

    device = args_cli.device
    model = build_model(args, device=device)
    load_ckpt_weights(model, ckpt, os.path.basename(args_cli.ckpt))
    vae = load_vae(args, device=device)

    from auto_eval_cpu import build_caches
    eval_cache, _, _ = build_caches(args, seen5_csv=None)
    if eval_cache is None:
        log("[large-eval] FATAL: cache 构建失败")
        return 1
    n = args_cli.n or len(eval_cache["conds"])
    log(f"[large-eval] n={n}, steps={args_cli.steps}, cfg={args_cli.cfg}, "
        f"seed={args_cli.seed}, batch={args_cli.batch}")

    from eval_auto import eval_gen_in_memory
    out_dir = os.path.join(args_cli.out_dir, args_cli.tag)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    mse, ssim = eval_gen_in_memory(
        model, vae, device, eval_cache,
        n=n, steps=args_cli.steps, cfg=args_cli.cfg, seed=args_cli.seed,
        batch=args_cli.batch,
        vis_out=os.path.join(out_dir, "grid.png"), vis_n=min(n, 16),
        cond_mode=getattr(args, "cond_mode", "2cond"),
        save_samples_dir=out_dir, step=None,
        glyph_init_mix=float(getattr(args, "glyph_init_mix", 0.0)))
    dt = time.time() - t0
    log(f"[large-eval] MSE={mse:.5f} SSIM={ssim:.4f} ({dt:.0f}s)")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": args_cli.tag, "ckpt": args_cli.ckpt, "csv": args_cli.csv,
                   "n": int(n), "mse": float(mse), "ssim": float(ssim),
                   "steps": args_cli.steps, "cfg": args_cli.cfg,
                   "seed": args_cli.seed, "seconds": round(dt, 1)},
                  f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
