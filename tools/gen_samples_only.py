#!/usr/bin/env python
"""只生成图、不算指标 —— 解耦，容错。

## 为什么
batch_eval 把「加载模型 -> 生成 -> 算指标」耦合在一起，任何一环失败整条就废。
而且失败原因多样（词表不匹配 / 架构不兼容 / eval 集缺 skel ...）。
先只把图生成出来落盘，之后对图片批量算指标（不依赖模型）。

## 输出
  assets/gen_samples/{run}__{step}__{set}/{idx:04d}_g.png / _gt.png
  以及 manifest.csv (run,step,set,idx,img_id,char,script,ok,err)

用法:
  CUDA_VISIBLE_DEVICES=0 python tools/gen_samples_only.py \
      --ckpt <path> --run <name> --csv assets/eval_v13_strict_fixed.csv --n 249
"""
import argparse
import csv
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

OUT_ROOT = "assets/gen_samples"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", default="")
    p.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv")
    p.add_argument("--n", type=int, default=249)
    p.add_argument("--cfg", type=float, default=0.7)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = get_args()
    run = a.run or os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(a.ckpt))))
    step = os.path.basename(a.ckpt)[:-3]
    setname = os.path.splitext(os.path.basename(a.csv))[0]
    out = os.path.join(OUT_ROOT, f"{run}__{step}__{setname}")
    os.makedirs(out, exist_ok=True)
    print(f"[gen] {run} step={step} -> {out}", flush=True)

    # ── 加载模型（用 model_io，它是唯一正确处理 Namespace 的路径）──
    from src.eval.model_io import load_model_from_ckpt
    from src.eval.inference import (make_eval_cache, load_eval_vae,
                                    sample_latents, build_diffusion)
    model, A = load_model_from_ckpt(a.ckpt, device=a.device, use_ema=True,
                                    verbose=False)
    vae = load_eval_vae(a.device, "data/pretrained/sd-vae-ft-ema")
    diff = build_diffusion(a.steps, "flow")

    # ── eval cache（这里最容易失败：词表 / skel 缺失）──────────────
    skel = (getattr(A, "eval_skel_latent_shards_dir", "") or
            getattr(A, "skel_latent_shards_dir", ""))
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))[:a.n]
    print(f"[gen] {len(rows)} 条, skel={skel}", flush=True)

    cmap = None
    csm = None
    try:
        if getattr(A, "callig_script_map", ""):
            from src.utils.callig_script_map import load_callig_script_map
            csm, _ = load_callig_script_map(A.callig_script_map)
        elif getattr(A, "callig_id_map", ""):
            from src.utils.callig_map import load_callig_id_map
            cmap, _ = load_callig_id_map(A.callig_id_map)
    except Exception as e:
        print(f"[gen] ⚠ 词表加载失败: {e}", flush=True)

    cache = make_eval_cache(a.csv, getattr(A, "img_root", None) or None,
                            skel, 32, len(rows), a.device,
                            callig_id_map=cmap, callig_script_map=csm,
                            vae=vae, force=True)
    gts, g_tok, cids = cache["gts"], cache["g_tok"], cache["cids"]
    print(f"[gen] cache ok: gts={tuple(gts.shape)}", flush=True)

    # ── 逐样本生成（每条独立 try，单条失败不影响其它）─────────────
    man = []
    sf = float(getattr(A, "vae_scaling_factor", 0.18215))
    for i, r in enumerate(rows):
        p = os.path.join(out, f"{i:04d}_gt.png")
        if not os.path.exists(p):
            Image.fromarray(((gts[i].detach().clamp(-1, 1) + 1) / 2
                             ).permute(1, 2, 0).numpy().mul(255).byte().numpy()
                            ).save(p)
        ok, err = 1, ""
        try:
            torch.manual_seed(a.seed + i)
            with torch.no_grad():
                lat = sample_latents(model, diff, 1, [i], cfg_scale=a.cfg,
                                     batch=1, device=a.device, cache=cache)
            rec = vae.decode(lat.to(a.device) / sf).sample.float().cpu()
            img = ((rec[0].detach().clamp(-1, 1) + 1) / 2
                   ).permute(1, 2, 0).numpy()
            Image.fromarray((img * 255).astype(np.uint8)).save(
                os.path.join(out, f"{i:04d}_g.png"))
        except Exception as e:
            ok, err = 0, f"{type(e).__name__}:{str(e)[:80]}"
            traceback.print_exc()
        man.append({"run": run, "step": step, "set": setname, "idx": i,
                    "img_id": r.get("old_50k_id", ""),
                    "char": r.get("character", ""),
                    "script": r.get("script", ""),
                    "callig": r.get("calligrapher", ""),
                    "ok": ok, "err": err})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    mp = os.path.join(OUT_ROOT, "manifest.csv")
    hdr = not os.path.exists(mp)
    with open(mp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        if hdr:
            w.writeheader()
        w.writerows(man)
    nok = sum(x["ok"] for x in man)
    print(f"\n[gen] ✓ {nok}/{len(man)} 成功 -> {out}", flush=True)
    print(f"[gen] manifest -> {mp}", flush=True)


if __name__ == "__main__":
    main()
