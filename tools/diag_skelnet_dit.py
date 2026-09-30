#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_skelnet_dit.py — 判定 clDice 为 0.025 是"没学会"还是"指标/空间对不齐"。

关键对照: **把输入的标准骨架直接当预测**(copy baseline)。
  · 若 copy baseline 的 clDice 明显高 (比如 >0.3), 说明指标与配准是对的, 模型只是还没学会;
  · 若 copy baseline 也接近 0, 说明是配准/口径问题 —— 再训也没有意义。

同时报墨量 (生成 / 输入 / GT), 判断是在造墨还是输出空白。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch as th
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
from tools.train_skelnet_dit import cldice_np           # noqa: E402

TIME_SCALE = 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="assets/skelnet_dit_v1.pt.best")
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-montage", default="")
    a = ap.parse_args()

    ck = th.load(a.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    ema = ck.get("ema") or ck["deform"]
    n_slots = int(ck.get("n_slots", 23))
    print(f"[ckpt] {a.ckpt} step={ck.get('step')} slots={n_slots} "
          f"best_clDice={ck.get('best_cldice')}")

    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=A["depth"], hidden_size=A["hidden"], num_heads=A["heads"],
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=A["inject_layers"], glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0, glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=A["style_drop"], cond_drop_one_prob=0.0,
        learn_sigma=False).to(a.device)
    miss = model.load_state_dict({k: v.to(a.device) for k, v in ema.items()},
                                strict=False)
    print(f"[ckpt] load missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
    model.eval()

    csmap = json.load(open(a.callig_map, encoding="utf-8")) if os.path.exists(a.callig_map) else None
    from src.utils.latent_dataset import MCCDLatentDataset
    ds = MCCDLatentDataset(csv_file=a.val_csv, latent_shards_dir=a.tgt_shards,
                           img_root="", image_size=256, is_train=False,
                           preload=True, load_image=False,
                           skel_latent_shards_dir=a.cond_shards,
                           callig_id_map=None, callig_script_map=csmap)
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(a.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    n = min(a.n, len(ds))
    cl_gen, cl_copy, ink_g, ink_s, ink_t, idio = [], [], [], [], [], []
    keep = []
    with th.no_grad():
        for s in range(0, n, 16):
            e = min(s + 16, n)
            xs, gs, ys, ids = [], [], [], []
            for i in range(s, e):
                b = ds[i]
                xs.append(b["latent"].float())
                gs.append(b["skel_latent"].float())
                ys.append(int(b["y_callig"]))
                ids.append(int(b["img_id"]))
            g = th.stack(gs).to(a.device)
            y = th.tensor(ys, device=a.device)
            z = th.randn_like(th.stack(xs).to(a.device))
            ts = th.linspace(1.0, 0.0, a.sample_steps + 1, device=a.device)
            for k in range(a.sample_steps):
                v = model(z, th.full((z.shape[0],), float(ts[k]) * TIME_SCALE,
                                     device=a.device),
                          y_callig=y, y_char=th.zeros_like(y), g=g)
                if isinstance(v, tuple):
                    v = v[0]
                z = z + (ts[k + 1] - ts[k]) * v
            pg = (vae.decode(z / 0.18215).sample.mean(1) < 0).cpu().numpy()
            ps = (vae.decode(g / 0.18215).sample.mean(1) < 0).cpu().numpy()
            for i in range(pg.shape[0]):
                fp = os.path.join(a.gt_png_dir, f"{ids[i]:06d}.png")
                if not os.path.exists(fp):
                    continue
                gt = np.asarray(Image.open(fp).convert("L")) < 128
                cl_gen.append(cldice_np(pg[i], gt))
                cl_copy.append(cldice_np(ps[i], gt))       # ★ copy baseline
                ink_g.append(float(pg[i].mean()))
                ink_s.append(float(ps[i].mean()))
                ink_t.append(float(gt.mean()))
                u = (pg[i] | ps[i]).sum()
                idio.append(float((pg[i] & ps[i]).sum()) / u if u else 0.0)
                keep.append((pg[i], ps[i], gt, ids[i]))
            print(f"    {e}/{n}", flush=True)

    print("================ 结果 ================")
    for tol in (0, 2, 4, 8):
        cg = float(np.mean([cldice_np(k[0], k[2], tol) for k in keep])) if keep else 0.0
        cc = float(np.mean([cldice_np(k[1], k[2], tol) for k in keep])) if keep else 0.0
        print(f"  tol={tol:>2}px  clDice(生成,GT) {cg:.4f}   clDice(输入std,GT) {cc:.4f}")
    print()
    f = lambda v: float(np.mean(v))
    print(f"  IoU(生成, 输入std)    {f(idio):.4f}")
    print(f"  墨量 生成 {f(ink_g):.4f} / 输入 {f(ink_s):.4f} / GT {f(ink_t):.4f}")
    print()
    cc4 = float(np.mean([cldice_np(k[1], k[2], 4) for k in keep])) if keep else 0.0
    cg4 = float(np.mean([cldice_np(k[0], k[2], 4) for k in keep])) if keep else 0.0
    if cc4 > 0.3:
        print(f"  => 指标口径可用 (copy baseline tol=4 为 {cc4:.3f})。"
              f"生成 {cg4:.3f} 与之相比即可判读。")
    else:
        print(f"  => ⚠ copy baseline 仍低 ({cc4:.3f}): GT 与 std 结构差异确实大, "
              f"或存在配准问题。")

    if a.save_montage and keep:
        rows = []
        for pg_, ps_, gt_, iid in keep[:8]:
            rows.append(np.concatenate([ps_, pg_, gt_], 1))
        m = np.concatenate([np.concatenate([r, np.ones((r.shape[0], 8), bool)], 1)
                            for r in rows], 0)
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(a.save_montage)
        print(f"  montage: {a.save_montage}  (列: 输入std | 生成 | GT)")


if __name__ == "__main__":
    main()
