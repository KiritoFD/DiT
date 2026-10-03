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
    # ★ 随机初始化基线: 不加载 ckpt 权重。用来判定探针的**判别力** ——
    #   若随机模型残差也只有 ~0.003 (与 100k 同量级), 说明这个度量无法区分好坏模型,
    #   之前"已经学很好"的结论作废。
    ap.add_argument("--init-random", action="store_true")
    ap.add_argument("--lp", type=int, default=2)
    a = ap.parse_args()
    dev = a.device

    from src.loss.flow_matching import TIME_SCALE
    from src.loss.structure_mid import blur2d, latent_dilate, lowpass, chan_norm

    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    import argparse as _ap
    args = ck.get("args", None)
    _ns = args if not isinstance(args, dict) else _ap.Namespace(**args)
    # 推理口径用 ema (与 eval 一致); 缺失则回退训练权重
    _use_ema = ("ema" in ck) and (ck["ema"] is not None)
    sd = ck["ema"] if _use_ema else ck.get("delta", ck.get("model", ck))
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    in_ch = sd["x_embedder.proj.weight"].shape[1]
    # ★ 不走 src.eval.model_io.load_model_from_ckpt: 它的 apply_post_construction 会
    #   断言 callig_emb_pretrained(45 行) 与 num_calligraphers=87(表 88 行) 配套 ——
    #   本 ckpt 里这两者不配套(预训练表是"连续索引 45 人") 。
    #   校准只需要模型能前向, 且下面用 strict=True **全量覆盖**
    #   权重(含 88 行表与 null_embed), 预训练表本来就无意义(会被覆盖),
    #   所以这里手动复刻 train.py 的后处理: freeze_table() 拆出 null_embed。
    from src.eval.model_io import build_model_from_args
    model = build_model_from_args(_ns, device=dev)
    model.y_callig_embedder.freeze_table()   # 拆出 null_embed, 对齐 ckpt key
    if a.init_random:
        print("[model] ★ --init-random: 保留随机初始化权重 (探针判别力基线)")
    else:
        model.load_state_dict(sd, strict=True)   # strict=True 当护栏
    model.eval()
    print(f"[model] {a.ckpt}  in_ch={in_ch}  ema={_use_ema}  strict=True OK "
          f"(params={sum(p.numel() for p in model.parameters()):,})")

    # 数据: 用 MCCDLatentDataset 取 N 条 (latent + g + y_callig + y_char)
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    ds = MCCDLatentDataset(
        csv_file=_ns.data_csv, latent_shards_dir=_ns.latent_shards_dir,
        img_root=getattr(_ns, "img_root", "") or "", image_size=256,
        # ★ skel_lat 只在 preload=True 时被填充 -> 非 preload 时 g=None, dilate 分支恒 inf
        preload=True, load_image=False,
        skel_latent_shards_dir=getattr(_ns, "skel_latent_shards_dir", "") or None,
        callig_id_map=None,
        callig_script_map=(_load_pair_map(_ns) if getattr(_ns, "callig_script_map", "") else None))
    idxs = list(range(min(a.n, len(ds))))
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs), num_workers=0)))
    x0 = batch["latent"].to(dev).float()[:, :int(_ns.latent_channels)]
    # ★ dataset 同时返回 'g' (未配置 glyph/inst 时是空张量 (B,0)) 与
    #   'skel_latent' (B,4,32,32) 。原写法 batch.get("g", batch.get("skel_latent")) 会
    #   优先拿到空的 'g' -> g=None -> dilate 分支恒 inf, 裁决失真。
    g = batch.get("skel_latent", None)
    if g is None or g.numel() == 0:
        g = batch.get("g", None)
    g = g.to(dev).float()[:, :int(_ns.latent_channels)] if g is not None and g.numel() else None
    _wants_g = bool(getattr(_ns, "skel_as_glyph_cond", False)) or \
        (float(getattr(_ns, "w_glyph_cond", 0) or 0) > 0) or \
        bool(getattr(_ns, "use_glyph_cond", False))
    print(f"[cond] skel_as_glyph_cond={getattr(_ns, 'skel_as_glyph_cond', None)} "
          f"w_glyph_cond={getattr(_ns, 'w_glyph_cond', None)} wants_g={_wants_g}")
    yc = batch["y_callig"].to(dev)
    yh = batch["y_char"].to(dev)
    print(f"[data] x0={tuple(x0.shape)} g={'None' if g is None else tuple(g.shape)}")

    tgrid = [float(x) for x in a.t_grid.split(",")]
    sgrid = [float(x) for x in a.sigma_grid.split(",")]
    kgrid = [float(x) for x in a.k_grid.split(",")]
    eps = th.randn_like(x0)                       # 固定噪声, 各 t 可比

    def resid(zp, zt):
        return float((lowpass(chan_norm(zp), a.lp) - lowpass(chan_norm(zt), a.lp)).pow(2).mean())

    out = {"t": [], "sigma": [], "k": [], "res_blur": [], "res_dil": []}
    print(f"\n{'t':>5} {'σ*':>6} {'res_blur':>9} | {'k*':>5} {'res_dil':>8} | carrier")
    with th.no_grad():
        for t in tgrid:
            x_t = (1 - t) * x0 + t * eps
            tt = th.full((x0.shape[0],), t, device=dev)
            mk = dict(y_callig=yc, y_char=yh)
            if g is not None and _wants_g:
                mk["g"] = g
            v = model(x_t, tt * TIME_SCALE, **mk)
            if isinstance(v, tuple):
                v = v[0]
            x0p = x_t - t * v[:, :x0.shape[1]]
            rb = [(resid(x0p, blur2d(x0, s)), s) for s in sgrid]
            print("    \u03c3-scan: " + "  ".join(f"{s:g}:{r:.4f}" for r, s in rb))
            rb = sorted(rb)
            res_b, sig = rb[0]
            if g is not None:
                rd = [(resid(x0p, latent_dilate(g, k)), k) for k in kgrid]
                print("    k-scan: " + "  ".join(f"{k:g}:{r:.4f}" for r, k in rd))
                rd = sorted(rd); res_d, kk = rd[0]
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
