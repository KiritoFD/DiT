#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mid_skel_iou.py — 直接回答两个问题 (不靠间接残差, 用**下游可解释指标**):

  Q1. 中程时间步的预测 ``x0_pred = E[x0|x_t]`` 解码回图后, 它的**骨架**到底学没学好?
      -> 算 skel IoU@k (骨架膨胀 k px 后的 IoU, 容忍细线难以严格重合) 与 clDice。
  Q2. 应该用多粗的骨架当载体?
      -> 统计 **GT 真实笔画宽度分布** (EDT: 骨架点处 2d+1), 给分位数。

为什么不再用 probe_coarse_skel.py 的"残差"
------------------------------------------
残差是 latent 域的均方距离, 量纲不可解释 (0.145 到底是好是坏?)。而且实测它
**几乎不随 t 变化**, 说明测到的主要是"骨架形态 vs 墨迹形态"的恒定失配, 不是
预测质量。IoU 是 0~1 的绝对量, 能直接说"学好了 / 没学好"。

用法 (远端 GPU):
  CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -u tools/mid_skel_iou.py \
      --ckpt <100k.pt> --t-grid 0.3,0.5,0.7 --n 256 \
      --dump-dir assets/mid_skel_dump --device cuda
"""
import argparse
import csv as _csv
import os
import re
import sys

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


# ── pixel 域形态学 ───────────────────────────────────────────────────────────
def _binary(gray_u8, thr=127):
    """白底黑字 -> 笔画 mask (True=笔画)."""
    return gray_u8 < thr


def _skeletonize():
    from skimage.morphology import skeletonize
    return skeletonize


def _widths_of(binary, skel):
    """骨架点处的局部笔画宽度 = 2*d + 1 (d=该点到最近背景的欧氏距离)。"""
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(binary)
    return 2.0 * d[skel] + 1.0


def _skel_iou(a, b, k=0):
    """骨架 IoU: 两份 1px 细线直接比必然 ~0, 故先各膨胀 k px 再比 (容忍小错位)。"""
    if k > 0:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        se = generate_binary_structure(2, 2)
        a = binary_dilation(a, structure=se, iterations=k)
        b = binary_dilation(b, structure=se, iterations=k)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(max(union, 1))


def _cldice(pred_mask, gt_mask, skel_fn):
    """clDice: 拓扑保持指标。Tp=预测骨架落在 GT 笔画内; Ts=GT 骨架落在预测笔画内。

    对"细线难严格重合"比裸 IoU 鲁棒, 且不需要调膨胀半径。
    """
    sk_p = skel_fn(pred_mask)
    sk_g = skel_fn(gt_mask)
    tp = float(np.logical_and(sk_p, gt_mask).sum()) / max(float(sk_p.sum()), 1.0)
    ts = float(np.logical_and(sk_g, pred_mask).sum()) / max(float(sk_g.sum()), 1.0)
    return 2.0 * tp * ts / max(tp + ts, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--t-grid", default="0.3,0.5,0.7")
    ap.add_argument("--iou-k", default="1,2,3")
    ap.add_argument("--vae-path", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--dec-batch", type=int, default=32)
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--dump-n", type=int, default=3)
    ap.add_argument("--out-json", default="assets/mid_skel_iou.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    from PIL import Image
    from src.loss.flow_matching import TIME_SCALE

    # ── 1) 模型 ──────────────────────────────────────────────────────────────
    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    _ns = ck["args"]
    sd = ck["ema"] if (("ema" in ck) and ck["ema"] is not None) else ck.get("delta", ck)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items() if isinstance(v, th.Tensor)}
    from src.eval.model_io import build_model_from_args
    model = build_model_from_args(_ns, device=dev)
    model.y_callig_embedder.freeze_table()
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[model] {a.ckpt}  strict=True OK")

    # ── 2) 数据 ──────────────────────────────────────────────────────────────
    import json as _json
    _csm = getattr(_ns, "callig_script_map", "") or ""
    csm = _json.load(open(_csm, encoding="utf-8")) if _csm and os.path.exists(_csm) else None
    from src.utils.latent_dataset import MCCDLatentDataset
    from torch.utils.data import DataLoader, Subset
    ds = MCCDLatentDataset(
        csv_file=_ns.data_csv, latent_shards_dir=_ns.latent_shards_dir,
        img_root="", image_size=256, preload=True, load_image=False,
        skel_latent_shards_dir=getattr(_ns, "skel_latent_shards_dir", "") or None,
        callig_id_map=None, callig_script_map=csm)
    idxs = list(range(min(a.n, len(ds))))
    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs), num_workers=0)))
    x0 = batch["latent"].to(dev).float()[:, :int(_ns.latent_channels)]
    gstd = batch.get("skel_latent", None)
    gstd = gstd.to(dev).float()[:, :int(_ns.latent_channels)] if gstd is not None and gstd.numel() else None
    yc, yh = batch["y_callig"].to(dev), batch["y_char"].to(dev)
    ids = [int(v) for v in batch["img_id"]]

    rows = list(_csv.DictReader(open(_ns.data_csv, encoding="utf-8")))
    path_of = {}
    for r in rows:
        m = re.search(r"(\d+)\.png", r["image_path"])
        if m:
            path_of[int(m.group(1))] = r["image_path"]
    print(f"[data] n={len(ids)}  x0={tuple(x0.shape)}")

    # ── 3) VAE ───────────────────────────────────────────────────────────────
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae_path).to(dev).eval()

    @th.no_grad()
    def _decode(z):
        """latent (B,4,32,32) [scaled] -> uint8 灰度 (B,256,256) 0..255 白底黑字。"""
        out = []
        for s in range(0, z.shape[0], a.dec_batch):
            x = vae.decode(z[s:s + a.dec_batch] / 0.18215).sample
            g = ((x * 0.5 + 0.5) * 255.0).clamp(0, 255).byte().float().mean(1)
            out.append(g)
        g = th.cat(out, 0).cpu().numpy().astype(np.uint8)
        return g

    # ── 4) Q2: GT 真实笔画宽度分布 ───────────────────────────────────────────
    _SKEL = _skeletonize()
    ws_all, gt_bin, gt_skel = [], [], []
    for iid in ids:
        arr = np.asarray(Image.open(path_of[iid]).convert("L"))
        b = _binary(arr); sk = _SKEL(b)
        gt_bin.append(b); gt_skel.append(sk)
        ws_all.append(_widths_of(b, sk))
    ws = np.concatenate(ws_all)
    qs = [10, 25, 50, 75, 90]
    wp = {str(q): float(np.percentile(ws, q)) for q in qs}
    print("\n[Q2] GT 真实笔画宽度分布 (px, 骨架点处 2d+1):")
    print(f"     mean={ws.mean():.1f}  " +
          "  ".join(f"p{q}={wp[str(q)]:.1f}" for q in qs))

    # ── 5) Q1: 各中程 t 的 x0_pred 解码 -> 骨架 IoU ──────────────────────────
    tgrid = [float(x) for x in a.t_grid.split(",")]
    ks = [int(x) for x in a.iou_k.split(",")]
    eps = th.randn_like(x0)
    _wants_g = bool(getattr(_ns, "skel_as_glyph_cond", False)) or \
        (float(getattr(_ns, "w_glyph_cond", 0) or 0) > 0)
    scaling = 0.18215

    res = {"t": tgrid, "iou_k": ks, "width": wp, "width_mean": float(ws.mean()),
           "rows": []}
    print(f"\n[Q1] 中程 x0_pred 解码后的结构指标 (skel IoU@k / clDice / mask IoU)")
    hdr = "  " + "".join(f"{'IoU@' + str(k):>9}" for k in ks) + f"{'clDice':>9}{'maskIoU':>9}"
    print("载体" + hdr)
    print("-" * (len(hdr) + 8))

    # 上界对照: GT latent 自己 decode 一遍 (VAE 重建误差 -> 不是完美 1.0)
    gt_dec = _decode(x0)
    _row = {"tag": "GT latent 重建 (上界)"}
    ious = []
    for k in ks:
        v = np.mean([_skel_iou(_SKEL(_binary(gt_dec[i])), gt_skel[i], k)
                     for i in range(len(ids))])
        ious.append(round(float(v), 4))
    _row["iou"] = ious
    _row["cldice"] = round(float(np.mean([_cldice(_binary(gt_dec[i]), gt_bin[i], _SKEL)
                                          for i in range(len(ids))])), 4)
    _row["mask_iou"] = round(float(np.mean([
        np.logical_and(_binary(gt_dec[i]), gt_bin[i]).sum() /
        max(np.logical_or(_binary(gt_dec[i]), gt_bin[i]).sum(), 1)
        for i in range(len(ids))])), 4)
    res["rows"].append(_row)
    print(f"{_row['tag']:<22}" + "".join(f"{v:>9.4f}" for v in _row["iou"]) +
          f"{_row['cldice']:>9.4f}{_row['mask_iou']:>9.4f}")

    dumps = {}
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
            dec = _decode(x0p)
            row = {"tag": f"x0_pred @ t={t:.2f}"}
            row["iou"] = [round(float(np.mean([_skel_iou(_SKEL(_binary(dec[i])), gt_skel[i], k)
                                               for i in range(len(ids))])), 4) for k in ks]
            row["cldice"] = round(float(np.mean([_cldice(_binary(dec[i]), gt_bin[i], _SKEL)
                                                 for i in range(len(ids))])), 4)
            row["mask_iou"] = round(float(np.mean([
                np.logical_and(_binary(dec[i]), gt_bin[i]).sum() /
                max(np.logical_or(_binary(dec[i]), gt_bin[i]).sum(), 1)
                for i in range(len(ids))])), 4)
            res["rows"].append(row)
            print(f"{row['tag']:<22}" + "".join(f"{v:>9.4f}" for v in row["iou"]) +
                  f"{row['cldice']:>9.4f}{row['mask_iou']:>9.4f}")
            if a.dump_dir:
                dumps[t] = dec

    # ── 6) dump 可视化: GT | t=... 横排 ──────────────────────────────────────
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
        for i in range(min(a.dump_n, len(ids))):
            panels = [np.asarray(Image.open(path_of[ids[i]]).convert("L"))]
            panels += [dumps[t][i] for t in tgrid]
            Image.fromarray(np.concatenate(panels, axis=1)).save(
                os.path.join(a.dump_dir, f"mid_{ids[i]}.png"))
        print(f"[dump] {min(a.dump_n, len(ids))} 张 -> {a.dump_dir} "
              f"(顺序: GT | " + " | ".join(f"t={t}" for t in tgrid) + ")")

    _json.dump(res, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[ok] -> {a.out_json}")


if __name__ == "__main__":
    main()
