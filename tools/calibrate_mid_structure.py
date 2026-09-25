#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calibrate_mid_structure.py — 一次前向校准中程结构 loss 的 σ*(t)/k*(t) 并裁决载体。

对应 docs 讨论: 别拍脑袋定 7px。用现成 ckpt, 对每个 t 采一批 x0_pred=E[x0|x_t],
在低通+通道归一子空间里, 分别对「blur(GT,σ)」和「dilate(skel,k)」做网格搜索取最小残差:
  σ*(t) = argmin_σ ||LP(cn(x0_pred)) - LP(cn(blur(GT,σ)))||
  k*(t) = argmin_k ||LP(cn(x0_pred)) - LP(cn(dilate(skel,k)))||
res_blur vs res_dil -> 哪个载体残差更小就选谁 (预期 blur(GT) 赢)。
输出 json 供 src/loss/structure_mid.py 的 TSchedule.from_json 直接吃。

用法 (远端 GPU):
  python tools/calibrate_mid_structure.py \
      --ckpt assets/results/v13_base_50k/*/checkpoints/0155000.pt \
      --t-grid 0.3,0.4,0.5,0.6,0.7 --n 256 \
      --out-json assets/std_mid_calib.json
"""
import argparse
import glob
import importlib.util as _ilu
import os
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--t-grid", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--sigma-grid", default="0,0.5,1,1.5,2,2.5,3,4")
    ap.add_argument("--k-grid", default="0,1,2,3,4")
    ap.add_argument("--out-json", default="assets/std_mid_calib.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from src.loss.flow_matching import TIME_SCALE
    from src.loss.structure_mid import blur2d, latent_dilate, lowpass, chan_norm

    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("delta", ck.get("model", ck))
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    args = ck.get("args", None)
    d = vars(args) if args is not None else {}
    in_ch = sd["x_embedder.proj.weight"].shape[1]
    _sp = _ilu.spec_from_file_location("_cfs", os.path.join(ROOT, "tools", "cfg_sweep.py"))
    _cfs = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_cfs)
    import argparse as _ap
    _ns = _ap.Namespace(**d)
    for _k, _v in (("image_size", 256), ("vae_downscale", 8), ("latent_channels", 4),
                   ("aux_latent_shards_dirs", ""), ("num_calligraphers", 87),
                   ("num_characters", 7765), ("callig_embed_dim", 128),
                   ("char_embed_dim", 384), ("condition_fusion", "factorized_cat")):
        if getattr(_ns, _k, None) is None:
            setattr(_ns, _k, _v)
    # ★ 改用项目自带的 load_model_from_ckpt —— 手搓 build+load_state_dict 会在
    #   strict=True 下报 "Unexpected key(s): y_callig_embedder.null_embed",
    #   因为该 key 只在 freeze_callig_table=True (freeze_table()) 时才存在。
    #   model_io 已处理: torch.compile 的 `_orig_mod.` 前缀 + freeze_table + strict 护栏。
    from src.eval.model_io import load_model_from_ckpt
    model, _ns = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=True)
    print(f"[model] {a.ckpt}  in_ch={in_ch}  loaded via src.eval.model_io (strict=True)")

    # 数据: 用 MCCDLatentDataset 取 N 条 (latent + g + y_callig + y_char)
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    ds = MCCDLatentDataset(
        csv_file=_ns.data_csv, latent_shards_dir=_ns.latent_shards_dir,
        img_root=getattr(_ns, "img_root", "") or "", image_size=256,
        preload=False, load_image=False,
        skel_latent_shards_dir=getattr(_ns, "skel_latent_shards_dir", "") or None,
        callig_id_map=None,
        callig_script_map=(_load_pair_map(_ns) if getattr(_ns, "callig_script_map", "") else None))
    idxs = list(range(min(a.n, len(ds))))
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs), num_workers=0)))
    x0 = batch["latent"].to(dev).float()[:, :int(_ns.latent_channels)]
    g = batch.get("g", batch.get("skel_latent", None))
    g = g.to(dev).float()[:, :int(_ns.latent_channels)] if g is not None and g.numel() else None
    yc = batch["y_callig"].to(dev)
    yh = batch["y_char"].to(dev)
    print(f"[data] x0={tuple(x0.shape)} g={'None' if g is None else tuple(g.shape)}")

    tgrid = [float(x) for x in a.t_grid.split(",")]
    sgrid = [float(x) for x in a.sigma_grid.split(",")]
    kgrid = [float(x) for x in a.k_grid.split(",")]
    eps = th.randn_like(x0)                       # 固定噪声, 各 t 可比

    def resid(zp, zt):
        return float((lowpass(chan_norm(zp)) - lowpass(chan_norm(zt))).pow(2).mean())

    out = {"t": [], "sigma": [], "k": [], "res_blur": [], "res_dil": []}
    print(f"\n{'t':>5} {'σ*':>6} {'res_blur':>9} | {'k*':>5} {'res_dil':>8} | carrier")
    with th.no_grad():
        for t in tgrid:
            x_t = (1 - t) * x0 + t * eps
            tt = th.full((x0.shape[0],), t, device=dev)
            mk = dict(y_callig=yc, y_char=yh)
            if g is not None:
                mk["g"] = g
            v = model(x_t, tt * TIME_SCALE, **mk)
            if isinstance(v, tuple):
                v = v[0]
            x0p = x_t - t * v[:, :x0.shape[1]]
            rb = [(resid(x0p, blur2d(x0, s)), s) for s in sgrid]
            rb.sort()
            res_b, sig = rb[0]
            if g is not None:
                rd = [(resid(x0p, latent_dilate(g, k)), k) for k in kgrid]
                rd.sort(); res_d, kk = rd[0]
            else:
                res_d, kk = float("inf"), 0
            out["t"].append(t); out["sigma"].append(sig); out["k"].append(kk)
            out["res_blur"].append(round(res_b, 5)); out["res_dil"].append(round(res_d, 5))
            print(f"{t:>5.2f} {sig:>6.2f} {res_b:>9.4f} | {kk:>5.0f} {res_d:>8.4f} | "
                  f"{'blur_gt' if res_b <= res_d else 'dilate_skel'}")

    carrier = "blur_gt" if np.mean(out["res_blur"]) <= np.mean(out["res_dil"]) else "dilate_skel"
    out["carrier"] = carrier
    import json
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[ok] 载体裁决: {carrier}  -> {a.out_json}")


def _load_pair_map(ns):
    import json
    p = getattr(ns, "callig_script_map", "")
    if p and os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


if __name__ == "__main__":
    main()
