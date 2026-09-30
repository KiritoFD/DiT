#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_predskel.py — 排查「predskel 解码发白」的根因 (GPU)。

检查链:
  [1] 三套骨架 latent 的统计量 + 解码墨迹率 (std / GT / 我生成的 pred)
  [2] 现场跑一遍 deform_skel(g_std, style), 与我存的 predskel 比 —— 是否一致
  [3] 检查模块内部开关: use_stroke_mod / topo_mode / z_bg / last_off 统计
  [4] 逐步定位: 关掉 stroke_mod / topo 后输出是否恢复
"""
import glob
import json
import os
import sys

import numpy as np
import torch

os.chdir("/root/Workspace/xy/DiT")
sys.path.insert(0, "/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SF = 0.18215
V24_CKPT = ("assets/results/v24_frozenskel/"
            "20260928-125010-v24-frozenskel/checkpoints/0090000.pt")


def ids(d):
    m = {}
    for f in glob.glob(d + "/*.npz"):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                m[int(i)] = z["latents"][j].astype(np.float32)
    return m


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    std = ids("data/top10_style23/shards_std")
    gt = ids("data/top10_style23/shards_aux_skel3")
    pr = ids("data/top10_style23/predskel_eval_seen20")
    print(f"shards_std={len(std)} aux_skel3={len(gt)} pred_seen20={len(pr)}")

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("pretrained_models/sd-vae-ft-ema").to(dev).eval()

    def dec(lat):
        with torch.no_grad():
            z = torch.from_numpy(lat[None]).to(dev) / SF
            g = vae.decode(z).sample[0].mean(0)
        return g.cpu().numpy()

    print("\n[1] 三套 latent 的统计 + 解码墨迹率")
    ks = sorted(pr)[:4]
    for i in ks:
        print(f"  id={i}")
        for tag, src in (("std", std.get(i)), ("gt", gt.get(i)), ("pred", pr.get(i))):
            if src is None:
                print(f"    {tag:<5} (缺)"); continue
            g = dec(src)
            print(f"    {tag:<5} lat absmean={np.abs(src).mean():.4f} "
                  f"min={src.min():+.2f} max={src.max():+.2f} | "
                  f"解码 mean={g.mean():.1f} min={g.min():+.2f} ink(<0)={ (g<0).mean():.4f}")

    # ---- [2] 现场重算 deform, 与存的 pred 比 ----
    print("\n[2] 现场 deform_skel(g_std, style) vs 存的 predskel")
    from src.eval.model_io import build_model_from_args
    ck = torch.load(V24_CKPT, map_location="cpu", weights_only=False)
    aa = ck.get("args")
    aa = aa if isinstance(aa, dict) else vars(aa)
    model = build_model_from_args(aa, dev)
    ds = model.deform_skel.to(dev).eval()
    st = torch.load("assets/callig_script_emb_top10.pt", map_location="cpu",
                    weights_only=False)
    tab = (st["embedding"] if isinstance(st, dict) else st).float().to(dev)
    pm = json.load(open("assets/callig_script_id_map_top10.json", encoding="utf-8"))["pair_map"]
    rows = {}
    import csv as _c
    for r in _c.DictReader(open("assets/eval_top10_seen_20.csv", encoding="utf-8")):
        rows[int(os.path.basename(r["image_path"])[:-4])] = r

    for i in ks[:2]:
        r = rows[i]
        pid = int(pm.get(f"{int(r['calligrapher_id'])}:{int(r['script_id'])}", 0))
        g = torch.from_numpy(std[i][None]).to(dev)
        s = tab[pid][None]
        with torch.no_grad():
            o = ds(g, s)
        o = o[0] if isinstance(o, (tuple, list)) else o
        onp = o[0].float().cpu().numpy()
        print(f"  id={i} pair_id={pid}")
        print(f"    现场 deform: absmean={np.abs(onp).mean():.4f} "
              f"min={onp.min():+.2f} max={onp.max():+.2f}")
        print(f"    存的 pred  : absmean={np.abs(pr[i]).mean():.4f} "
              f"min={pr[i].min():+.2f} max={pr[i].max():+.2f}")
        print(f"    两者 max|Δ| = {np.abs(onp - pr[i]).max():.6f}")
        gdec = dec(onp)
        print(f"    现场 deform 解码: mean={gdec.mean():.1f} ink(<0)={(gdec<0).mean():.4f}")

    # ---- [3] 模块内部开关 ----
    print("\n[3] DeformSkel 内部开关")
    for k in ("use_stroke_mod", "stroke_conv", "stroke_style", "z_bg", "delta_ink",
              "topo_mode", "prune", "lig", "dt_ch", "blur", "residual", "res"):
        v = getattr(ds, k, "<无此属性>")
        if isinstance(v, torch.Tensor):
            v = f"tensor{tuple(v.shape)}"
        elif v is not None and not isinstance(v, (int, float, bool, str)):
            v = type(v).__name__
        print(f"  {k:<16} = {v}")
    print(f"  last_off absmean = {getattr(ds,'last_off',None).abs().mean().item() if getattr(ds,'last_off',None) is not None else None}")
    print(f"  offset_stats = {ds.offset_stats()}")


if __name__ == "__main__":
    main()
