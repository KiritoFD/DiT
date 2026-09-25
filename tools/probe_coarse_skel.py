#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_coarse_skel.py — 「粗骨架」中程载体的**一次性探针**（小样本, 全内存, 不落盘）。

为什么要有这个脚本
------------------
`calibrate_mid_structure.py` 里的 dilate_skel 分支用的是 **latent 域 max-pool 近似**
去膨胀"标准字形细骨架"(g), 实测残差 0.49 —— 比 blur_gt(0.0024) 差 200 倍。
但那个数字混杂了三件事, 不能直接拿来判"骨架路线死刑":

  1. 粗细不对: 1px 细骨架 vs 中程软墨迹, 本来就差得远 (中程形态是"带宽度包络的笔画")
  2. 来源不对: g 是**标准字形**, 而 x0_pred 预测的是**这个书家实际写的字**
  3. 膨胀方式不对: latent 域 max-pool ≠ pixel 域形态学膨胀再编码 (VAE 非线性)

本脚本把这三件事**分开**: 直接从 **GT 图**(实例来源) 提 1px 骨架, 用 **EDT 精确加粗**
到目标像素宽度, 再 **VAE encode** 成 latent, 最后在同一批样本上测
`||LP(cn(x0_pred)) - LP(cn(coarse_lat))||²`。

输出一张 width × t 的残差表 + 对照行 (GT 本身 / 标准字形细骨架 g), 一眼看出
"粗骨架到底是不是比细骨架更接近中程预测形态"。

用法 (远端 GPU):
  CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -u tools/probe_coarse_skel.py \
      --ckpt <100k.pt> --widths 1,3,6,10,14 --n 256 --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _thicken(binary, width):
    """1px 骨架 -> 精确 width 像素宽的粗骨架 (EDT 半径法, 比迭代膨胀精确且快)。

    distance_transform_edt(~binary) 给出每个**背景**像素到最近骨架像素的欧氏距离;
    保留 d <= (width-1)/2 即得到总宽 ≈ width 的笔画带。
    """
    if width <= 1:
        return binary
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~binary)
    return d <= (float(width) - 1.0) / 2.0


def _skeletonize():
    """复用 tools/build_skel_latents.py 的实现 (含无 skimage 时的 fallback)。"""
    import importlib.util as _ilu
    sp = _ilu.spec_from_file_location(
        "_bsl", os.path.join(ROOT, "tools", "build_skel_latents.py"))
    m = _ilu.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m._skel_impl()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--widths", default="1,3,6,10,14")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--t-grid", default="0.3,0.5,0.7")
    ap.add_argument("--lp", type=int, default=2)
    ap.add_argument("--vae-path", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--vae-batch", type=int, default=64)
    ap.add_argument("--out-json", default="assets/coarse_skel_probe.json")
    ap.add_argument("--dump-dir", default="",
                    help="把样例横排存图: GT | 各宽度骨架, 用于肉眼确认'N px 到底多粗'")
    ap.add_argument("--dump-n", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from PIL import Image
    from src.loss.flow_matching import TIME_SCALE
    from src.loss.structure_mid import lowpass, chan_norm

    # ── 1) 模型 (与 calibrate_mid_structure.py 同一套严格加载) ──────────────
    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    _ns = ck["args"]
    _use_ema = ("ema" in ck) and (ck["ema"] is not None)
    sd = ck["ema"] if _use_ema else ck.get("delta", ck.get("model", ck))
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    from src.eval.model_io import build_model_from_args
    model = build_model_from_args(_ns, device=dev)
    model.y_callig_embedder.freeze_table()
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[model] {a.ckpt}  ema={_use_ema}  strict=True OK")

    # ── 2) 数据 (preload=True 才有 skel_latent) ──────────────────────────────
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    # ★ 必须传 callig_script_map: 它把 (书家id, 书体id) 映射到 [0,87] 的连续索引。
    #   传 None 会让 y_callig 直接用原始 calligrapher_id (可达 591) -> 查表越界
    #   -> CUDA device-side assert (不是 Python IndexError, 异步报得晚, 很难查)。
    import json as _json
    _csm = getattr(_ns, "callig_script_map", "") or ""
    csm = _json.load(open(_csm, encoding="utf-8")) if _csm and os.path.exists(_csm) else None
    print(f"[data] callig_script_map = {_csm or '(None)'}")
    ds = MCCDLatentDataset(
        csv_file=_ns.data_csv, latent_shards_dir=_ns.latent_shards_dir,
        img_root="", image_size=256, preload=True, load_image=False,
        skel_latent_shards_dir=getattr(_ns, "skel_latent_shards_dir", "") or None,
        callig_id_map=None,
        callig_script_map=csm)
    idxs = list(range(min(a.n, len(ds))))
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs), num_workers=0)))
    x0 = batch["latent"].to(dev).float()[:, :int(_ns.latent_channels)]
    gstd = batch.get("skel_latent", None)
    gstd = gstd.to(dev).float()[:, :int(_ns.latent_channels)] if gstd is not None and gstd.numel() else None
    yc, yh = batch["y_callig"].to(dev), batch["y_char"].to(dev)
    ids = [int(v) for v in batch["img_id"]]
    print(f"[data] x0={tuple(x0.shape)}  std_skel(g)={None if gstd is None else tuple(gstd.shape)}  n={len(ids)}")

    # ── 3) GT 图 -> 1px 骨架 -> 各宽度加粗 (CPU, 多进程可后加; 256 张直接串行够快) ──
    _SKEL = _skeletonize()
    widths = [int(w) for w in a.widths.split(",")]
    coarse = {w: [] for w in widths}          # width -> list[(B,256,256) uint8]
    img_root = os.path.dirname(_ns.data_csv and "")
    # GT 图路径: 直接用 csv 的 image_path 列 (与 dataset 同源, 保证 id 对齐)
    import csv as _csv
    rows = list(_csv.DictReader(open(_ns.data_csv, encoding="utf-8")))
    path_of = {}
    for r in rows:
        _m = __import__("re").search(r"(\d+)\.png", r["image_path"])
        if _m:
            path_of[int(_m.group(1))] = r["image_path"]
    miss = 0
    for iid in ids:
        p = path_of.get(iid)
        if p is None or not os.path.exists(p):
            miss += 1
            for w in widths:
                coarse[w].append(np.full((256, 256), 255, dtype=np.uint8))
            continue
        arr = np.asarray(Image.open(p).convert("L"))
        skel1 = _SKEL(arr < 127)              # 白底黑字: 暗像素 = 笔画
        for w in widths:
            m = _thicken(skel1, w)
            coarse[w].append(np.where(m, 0, 255).astype(np.uint8))   # 白底黑线
    print(f"[skel] GT 图解析完成 (missing={miss}/{len(ids)})")

    # ── 4) VAE encode ────────────────────────────────────────────────────────
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae_path).to(dev).eval()
    print(f"[vae] loaded {a.vae_path}")

    def _encode(stack_u8):
        out = []
        with th.no_grad():
            for s in range(0, len(stack_u8), a.vae_batch):
                imgs = np.stack(stack_u8[s:s + a.vae_batch])
                x = th.from_numpy(imgs.astype(np.float32)) / 255.0 * 2.0 - 1.0
                x = x.unsqueeze(1).repeat(1, 3, 1, 1).to(dev)
                lat = vae.encode(x).latent_dist.sample() * 0.18215
                out.append(lat.float())
        return th.cat(out, 0)

    coarse_lat = {w: _encode(coarse[w]) for w in widths}
    for w in widths:
        print(f"[vae] width={w}px -> {tuple(coarse_lat[w].shape)}")

    # ── 4.5) 肉眼确认: 各宽度骨架到底多粗 (横排 GT | w1 | w2 | ...) ──────────
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        for k in range(min(a.dump_n, len(ids))):
            iid = ids[k]
            panels = [np.asarray(Image.open(path_of[iid]).convert("L"))]
            for w in widths:
                panels.append(coarse[w][k])
            row = np.concatenate(panels, axis=1)
            Image.fromarray(row).save(os.path.join(a.dump_dir, f"sample_{iid}.png"))
        print(f"[dump] {min(a.dump_n, len(ids))} 张样例 -> {a.dump_dir} "
              f"(顺序: GT | " + " | ".join(f"{w}px" for w in widths) + ")")

    # ── 5) 探针: 各 t 下 x0_pred 与各类载体的残差 ────────────────────────────
    tgrid = [float(x) for x in a.t_grid.split(",")]
    eps = th.randn_like(x0)

    def resid(zp, zt):
        return float((lowpass(chan_norm(zp), a.lp) - lowpass(chan_norm(zt), a.lp)).pow(2).mean())

    _wants_g = bool(getattr(_ns, "skel_as_glyph_cond", False)) or \
        (float(getattr(_ns, "w_glyph_cond", 0) or 0) > 0)
    print(f"\n[cond] wants_g={_wants_g} (g=标准字形细骨架, 作为模型条件传入)")
    hdr = "  " + "".join(f"{'t=' + format(t, '.2f'):>12}" for t in tgrid)
    print("\n载体" + hdr)
    print("-" * (len(hdr) + 6))

    res = {"widths": widths, "t": tgrid, "coarse": {}, "gt": [], "std_skel_g": []}
    with th.no_grad():
        for t in tgrid:
            x_t = (1 - t) * x0 + t * eps
            tt = th.full((x0.shape[0],), t, device=dev)
            mk = dict(y_callig=yc, y_char=yh)
            if gstd is not None and _wants_g:
                mk["g"] = gstd
            v = model(x_t, tt * TIME_SCALE, **mk)
            v = v[0] if isinstance(v, tuple) else v
            x0p = x_t - t * v[:, :x0.shape[1]]
            res["gt"].append(round(resid(x0p, x0), 5))
            if gstd is not None:
                res["std_skel_g"].append(round(resid(x0p, gstd), 5))
            for w in widths:
                res["coarse"].setdefault(str(w), []).append(
                    round(resid(x0p, coarse_lat[w]), 5))

    print(f"{'GT 本身 (blur σ=0)':<22}" + "".join(f"{v:>12.5f}" for v in res["gt"]))
    if gstd is not None:
        print(f"{'标准字形细骨架 g':<22}" + "".join(f"{v:>12.5f}" for v in res["std_skel_g"]))
    for w in widths:
        tag = f"实例粗骨架 {w}px"
        print(f"{tag:<22}" + "".join(f"{v:>12.5f}" for v in res["coarse"][str(w)]))

    import json
    json.dump(res, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    best_w = min(widths, key=lambda w: float(np.mean(res["coarse"][str(w)])))
    print(f"\n[ok] 实例粗骨架最优宽度 = {best_w}px "
          f"(均值残差 {np.mean(res['coarse'][str(best_w)]):.5f})  -> {a.out_json}")


if __name__ == "__main__":
    main()
