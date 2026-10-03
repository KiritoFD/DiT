#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster_skelnet_now.py — 给「正在训练的 SkelNet」出骨架 poster + mse/l1/Dice。

列: std(g) | gen b=1 | gen b=0.634 | GT(w7 训练目标 latent) | GT(png, 3px 真迹骨架)
行: 每行一个样本 (默认留出字符验证集 assets/val_skelnet.csv, --stride 跨字抽样)

为什么并列 b=1 与 b=0.634: 之前的 poster/ink 都是在 b=0.634 **混合后**的 shards
上量的, 而训练期评估(纯生成器输出)报的是相反方向。并列才能判"变淡/变粗是谁造成的"。

为什么并列 w7 latent 与 3px png: 模型学的是 **7px 粗载体**。若 gen 与 w7 目标长得像,
说明"粗"是目标属性而不是模型糊了; 若不像, 才是模型的问题。

⚠ 渲染必须用原生 256 (--cell 256): 缩到 150 会**抽样丢掉 1px 笔画**, 让人误判为
  "严重碎片/丢笔画"(2026-09-30 踩过)。

用法:
  python tools/poster_skelnet_now.py --resume assets/skelnet_dit_I_w7_wdm.pt --n 6 --stride 400
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TIME_SCALE = 1000.0
STD_DIRS = ["data/top10_style23/shards_std", "data/50k_v2_glyph15k/shards_std"]


def load_index(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="assets/skelnet_dit_I_w7_wdm.pt")
    ap.add_argument("--csv", default="assets/val_skelnet.csv")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1,
                    help="候选里每隔 stride 取一条 -> 跨字符抽样 (val csv 相邻行常常同字)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.634)
    ap.add_argument("--std-dir", default="",
                    help="优先使用的标准骨架 latent 目录 (K 臂用 shards_std_w7, 5px)")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w7",
                    help="训练目标 (w7 粗载体) 的 latent shards, 用于判'粗是目标还是模型糊'")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--inject-layers", type=int, default=2)
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--out", default="_ot_scratch/poster_skelnet_now.png")
    ap.add_argument("--cell", type=int, default=256,
                    help="★ 默认 256 = 原生尺寸, 不缩放 (缩到 150 会抽样丢掉 1px 笔画)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    # ── 索引 (--std-dir 优先) ──
    dirs = ([a.std_dir] if a.std_dir else []) + STD_DIRS
    idxs = {}
    for d in dirs:
        if os.path.isdir(d):
            idxs[d] = load_index(d)
    tgt_idx = load_index(a.tgt_shards) if os.path.isdir(a.tgt_shards) else {}

    rows = []
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        m = re.search(r"(\d+)\.png$", r.get("image_path", ""))
        if not m:
            continue
        iid = int(m.group(1))
        if not os.path.exists(os.path.join(a.gt_png_dir, f"{iid:06d}.png")):
            continue
        if tgt_idx and iid not in tgt_idx:
            continue
        rows.append((iid, r))
    print(f"[1] {a.csv}: {len(rows)} 条候选 (有 GT png 且命中 w7 目标)")
    print("    std 目录命中: " + ", ".join(
        f"{os.path.basename(os.path.dirname(d))}={sum(1 for i, _ in rows if i in idxs[d])}"
        for d in idxs))

    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    from src.utils.callig_script_map import map_callig_script as _mcs
    from src.model.dit import DiT_2Cond
    from diffusers.models import AutoencoderKL
    from PIL import Image, ImageDraw

    n_slots = int(csmap.get("num_pairs", 0) or 0)
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        depth=a.depth, hidden_size=a.hidden, num_heads=a.heads,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=a.inject_layers, glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0, glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=0.1, cond_drop_one_prob=0.0, learn_sigma=False,
    ).to(dev)
    st = th.load(a.style_emb, map_location="cpu", weights_only=False)
    emb = st["embedding"] if isinstance(st, dict) else st
    w = model.y_callig_embedder.embedding_table.weight
    if tuple(emb.shape) == (w.shape[0] - 1, w.shape[1]):
        with th.no_grad():
            w[:emb.shape[0]].copy_(emb.float())
    rf = th.load(a.resume, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(dev) for k, v in rf["ema"].items()})
    model.eval()
    print(f"[2] 载入 {a.resume} (训练 step={rf.get('step')})")

    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ── 抽样 ──
    sel = []
    for iid, r in rows:
        src = next((d for d in dirs if d in idxs and iid in idxs[d]), None)
        if src is not None:
            sel.append((iid, r, src))
    sel = sel[::max(a.stride, 1)][:a.n]
    print(f"[3] 选中 {len(sel)} 条:")
    for iid, r, _ in sel:
        print(f"    id{iid}  {r.get('char') or '?'} / {r.get('calligrapher') or '?'}")

    gs, ys, ids, gts, tgts = [], [], [], [], []

    def _get(idx, iid):
        f, j = idx[iid]
        with np.load(f) as z:
            return np.asarray(z["latents"][j], np.float32)

    for iid, r, src in sel:
        gs.append(_get(idxs[src], iid))
        tgts.append(_get(tgt_idx, iid) if tgt_idx else np.zeros_like(gs[-1]))
        ys.append(int(_mcs(int(r["calligrapher_id"]), int(r["script_id"]), csmap)))
        ids.append(iid)
        gts.append(np.asarray(Image.open(
            os.path.join(a.gt_png_dir, f"{iid:06d}.png")).convert("L")) < 128)

    g = th.from_numpy(np.stack(gs)).to(dev)
    y = th.tensor(ys, dtype=th.long, device=dev)
    with th.no_grad():
        z = g.clone()
        ts = th.linspace(1.0, 0.0, a.steps + 1, device=dev)
        for s in range(a.steps):
            o = model(z, th.full((z.shape[0],), float(ts[s]) * TIME_SCALE, device=dev),
                      y_callig=y, y_char=th.zeros_like(y), g=th.zeros_like(g))
            if isinstance(o, tuple):
                o = o[0]
            z = z + (ts[s + 1] - ts[s]) * o
        z_raw, z_b = z.clone(), (1 - a.beta) * g + a.beta * z
        tgt_t = th.from_numpy(np.stack(tgts)).to(dev)

        def dec_bin(lat):
            return (vae.decode(lat / 0.18215).sample.mean(1) < 0).cpu().numpy()

        p_raw, p_b, p_std, p_tgt = (dec_bin(z_raw), dec_bin(z_b), dec_bin(g),
                                    dec_bin(tgt_t))

    print("\n[4] 指标 (二值, 墨=1; vs 3px GT png; mse/l1 对二值等价, 越低越好):")
    for nm, arr in (("gen b=1", p_raw), ("gen b=0.634", p_b), ("std(g)", p_std),
                    ("GT(w7 目标)", p_tgt)):
        ms, dc, iou, ink = [], [], [], []
        for i in range(len(sel)):
            p = arr[i].astype(np.float32)
            t = gts[i].astype(np.float32)
            ms.append(float(((p - t) ** 2).mean()))
            inter = float((p * t).sum())
            dc.append(2 * inter / max(float(p.sum() + t.sum()), 1e-9))
            iou.append(inter / max(float(((p + t) > 0).sum()), 1e-9))
            ink.append(float(p.mean()) / max(float(t.mean()), 1e-9))
        print(f"    {nm:14} mse {np.mean(ms):.5f}  Dice {np.mean(dc):.4f}  "
              f"IoU {np.mean(iou):.4f}  ink/GT {np.mean(ink):.3f}")

    # ── 渲染 (默认原生 256, 不缩放) ──
    cols = [("std(g)", p_std), ("gen b=1", p_raw), ("gen b=0.634", p_b),
            ("GT(w7 tgt)", p_tgt), ("GT(png 3px)", None)]
    cell, lab = a.cell, 26
    W = cell * len(cols) + 8
    H = lab * 2 + (cell + lab) * len(sel)
    canvas = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 6), "SkelNet poster (training model) | mse vs GT(3px) under each cell",
            fill=0)
    for c, (nm, _) in enumerate(cols):
        dr.text((c * cell + 6, lab + 4), nm, fill=0)
    for i in range(len(sel)):
        yy = lab * 2 + i * (cell + lab)
        for c, (nm, arr) in enumerate(cols):
            if arr is None:
                bits, mse = gts[i], float("nan")
            else:
                bits = arr[i]
                mse = float(((bits.astype(np.float32) - gts[i]) ** 2).mean())
            arr8 = np.where(bits, 0, 255).astype(np.uint8)
            im = (Image.fromarray(arr8) if cell == arr8.shape[-1]
                  else Image.fromarray(arr8).resize((cell, cell), Image.LANCZOS))
            canvas.paste(im, (c * cell, yy))
            txt = f"id{ids[i]}" + (f"  mse {mse:.3f}" if mse == mse else "")
            dr.text((c * cell + 4, yy + cell + 2), txt, fill=0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    canvas.save(a.out)
    print(f"[5] poster -> {a.out}  ({W}x{H})")


if __name__ == "__main__":
    main()
