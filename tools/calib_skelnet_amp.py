#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""calib_skelnet_amp.py — SkelNet-DiT 输出的**幅度校准**。

背景 (2026-09-30 链式实验结论):
  · G_bridge_nog (g起步+不给网络看g) resAlign=0.5067 ≈ 理论上限 0.5087 (99.6%)
    —— **残差方向几乎完美**;
  · 但 ink比 ≈ 1.44~1.59 (过墨 ~50%) -> clDice 仍低于 copy 基线;
  · cosGen 0.870 > cosStd 0.826 (唯一打败"照抄"的 arm)。
  => 剩下的问题疑似只剩**一个标量**: 残差放多大。

做法
----
对验证集 (assets/val_skelnet.csv, 按字符留出) 生成 z_1 (raw, β=1), 再构造
    z_β = g + β * (z_1 - g)
扫描 β 并解**每张样本的最优 β***:
    β*_i = <z_1-g, x0-g> / ||z_1-g||^2     (最小二乘, 闭式)
报告:
  · β* 分布 (p10/p50/p90) 与 ||z1-g||/||x0-g|| (幅度比)
  · 各 β 下的 clDice / idIoU / ink比 / cosGen / resAlign
  · copy(β=0) 基线对照

resAlign 对 β 不变 (方向不变), 所以 β 影响的是 clDice/ink/cosGen。

用法:
  python tools/calib_skelnet_amp.py --resume assets/skelnet_dit_G_bridge_nog.pt
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TIME_SCALE = 1000.0


def _skel(b):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(b)
    except Exception:                                          # noqa: BLE001
        from scipy.ndimage import binary_erosion, generate_binary_structure
        st = generate_binary_structure(2, 2)
        sk = np.zeros_like(b)
        cur = b.copy()
        while cur.any():
            er = binary_erosion(cur, structure=st)
            sk |= cur & ~er
            cur = er
        return sk


def cldice_np(pred_b, gt_b, tol=3):
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    from scipy.ndimage import binary_dilation
    sp, sg = _skel(pred_b), _skel(gt_b)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    st = np.ones((3, 3), bool)
    gt_t = binary_dilation(gt_b, structure=st, iterations=tol)
    pr_t = binary_dilation(pred_b, structure=st, iterations=tol)
    tprec = float((sp & gt_t).sum()) / float(sp.sum())
    tsens = float((sg & pr_t).sum()) / float(sg.sum())
    if tprec + tsens <= 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--resume", default="assets/skelnet_dit_G_bridge_nog.pt")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--inject-layers", type=int, default=2)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--betas", default="0,0.3,0.5,0.7,0.85,1.0")
    a = ap.parse_args()
    dev = "cuda"

    # ── 数据 (与 train_skelnet_dit.py 完全同口径) ─────────────────────────
    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    ds_va = MCCDLatentDataset(
        csv_file=a.val_csv, latent_shards_dir=a.tgt_shards, img_root="",
        image_size=256, is_train=False, preload=True, load_image=False,
        skel_latent_shards_dir=a.cond_shards,
        callig_id_map=None, callig_script_map=csmap)
    n_slots = int(csmap.get("num_pairs", 0) or 0)
    print(f"[1] 验证集 {len(ds_va)} 条, 风格槽位 {n_slots}")

    # ── 模型 ─────────────────────────────────────────────────────────────
    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=a.depth, hidden_size=a.hidden, num_heads=a.heads,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=a.inject_layers, glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0,
        glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=0.1, cond_drop_one_prob=0.0,
        learn_sigma=False,
    ).to(dev)
    rf = th.load(a.resume, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(dev) for k, v in rf["ema"].items()})
    model.eval()
    print(f"[2] 载入 {a.resume} (step={rf.get('step')})")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()

    def load_batch(idxs):
        xs, gs, ys, ids = [], [], [], []
        for i in idxs:
            b = ds_va[i]
            xs.append(b["latent"].float())
            gs.append(b["skel_latent"].float())
            ys.append(int(b["y_callig"]))
            ids.append(int(b["img_id"]))
        return (th.stack(xs).to(dev), th.stack(gs).to(dev),
                th.tensor(ys, device=dev), ids)

    def _cos(x, y):
        x, y = x.ravel().float(), y.ravel().float()
        return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))

    betas = [float(s) for s in a.betas.split(",")]
    n = min(a.val_n, len(ds_va))
    B = 32
    # 收集: z1(β=1 raw), g, x0, y, ids, gt png mask
    Z1, G, X0, Y, IDS, GTM, STM = [], [], [], [], [], [], []
    with th.no_grad():
        for s in range(0, n, B):
            end = min(s + B, n)
            x0, g, y, ids = load_batch(list(range(s, end)))
            z = g.clone()                                       # ★ g 起步
            ts = th.linspace(1.0, 0.0, a.sample_steps + 1, device=dev)
            for k in range(a.sample_steps):
                # ★ bridge-hide-g: 不把 g 喂给网络 (与训练一致)
                out = model(z, th.full((z.shape[0],), float(ts[k]) * TIME_SCALE,
                                       device=dev),
                            y_callig=y, y_char=th.zeros_like(y),
                            g=th.zeros_like(g))
                if isinstance(out, tuple):
                    out = out[0]
                z = z + (ts[k + 1] - ts[k]) * out
            gdec = vae.decode(g / 0.18215).sample.mean(1)
            st = (gdec < 0).cpu().numpy()
            from PIL import Image
            gtm = []
            for i in ids:
                fp = os.path.join(a.gt_png_dir, f"{i:06d}.png")
                gtm.append(np.asarray(Image.open(fp).convert("L")) < 128)
            Z1.append(z.cpu()); G.append(g.cpu()); X0.append(x0.cpu())
            Y.append(y.cpu()); IDS += list(ids)
            GTM += gtm; STM.append(st)
    Z1, G, X0 = th.cat(Z1), th.cat(G), th.cat(X0)
    STM = np.concatenate(STM)
    print(f"[3] 生成完成 {Z1.shape[0]} 张")

    # ── β* 闭式解 ────────────────────────────────────────────────────────
    d_gen = (Z1 - G).reshape(Z1.shape[0], -1)                   # 模型残差
    d_gt = (X0 - G).reshape(X0.shape[0], -1)                    # 真残差
    num = (d_gen * d_gt).sum(1)
    den = (d_gen * d_gen).sum(1).clamp_min(1e-8)
    bstar = (num / den).numpy()
    amp = (d_gen.norm(dim=1) / d_gt.norm(dim=1).clamp_min(1e-8)).numpy()
    print(f"[4] β* 分布: p10={np.percentile(bstar,10):.3f} p50={np.percentile(bstar,50):.3f} "
          f"p90={np.percentile(bstar,90):.3f} mean={bstar.mean():.3f} std={bstar.std():.3f}")
    print(f"    幅度比 ||z1-g||/||x0-g||: p10={np.percentile(amp,10):.2f} "
          f"p50={np.percentile(amp,50):.2f} p90={np.percentile(amp,90):.2f}")

    # resAlign (β 不变, 算一次)
    resal = [(_cos(d_gen[i], d_gt[i])) for i in range(Z1.shape[0])]
    print(f"    resAlign={np.mean(resal):.4f} (基线0/上限~0.51)")

    # ── 各 β 的像素域指标 ────────────────────────────────────────────────
    rows = []
    for beta in betas + [float(np.clip(np.median(bstar), 0, 1))]:
        tag = f"β*={beta:.3f}" if abs(beta - float(np.clip(np.median(bstar), 0, 1))) < 1e-6 else f"β={beta:.2f}"
        cls, idg, inkp, inkg, cg = [], [], [], [], []
        with th.no_grad():
            for s in range(0, Z1.shape[0], B):
                zb = (G[s:s+B] + beta * (Z1[s:s+B] - G[s:s+B])).to(dev)
                dec = vae.decode(zb / 0.18215).sample.mean(1)
                pr = (dec < 0).cpu().numpy()
                for i in range(pr.shape[0]):
                    gi = s + i
                    gt = GTM[gi]
                    cls.append(cldice_np(pr[i], gt))
                    u = (pr[i] | STM[gi]).sum()
                    idg.append(float((pr[i] & STM[gi]).sum()) / u if u else 0.0)
                    inkp.append(float(pr[i].mean()))
                    inkg.append(float(gt.mean()))
                    cg.append(_cos(zb[i].cpu(), X0[s+i]))
        rows.append((tag, beta, np.mean(cls), np.mean(idg),
                     np.mean(inkp) / max(np.mean(inkg), 1e-9),
                     np.mean(cg), np.mean(resal)))

    print(f"\n{'β':>10} {'clDice':>8} {'idIoU':>8} {'ink比':>7} {'cosGen':>8} {'resAlign':>9}")
    for tag, b_, c, i_, ir, cg_, ra in rows:
        print(f"{tag:>10} {c:8.4f} {i_:8.4f} {ir:7.3f} {cg_:8.4f} {ra:9.4f}")
    print(f"\n  copy 基线 (β=0, 直接用标准骨架): clDice 第一行")
    print(f"  β=1 (raw 模型输出): 最后一行 β=1.00")
    out_json = {"resume": a.resume, "step": rf.get("step"),
                "beta_star": {"p10": float(np.percentile(bstar, 10)),
                              "p50": float(np.percentile(bstar, 50)),
                              "p90": float(np.percentile(bstar, 90)),
                              "mean": float(bstar.mean())},
                "amp_ratio": {"p50": float(np.percentile(amp, 50))},
                "rows": [{"tag": t, "beta": b, "clDice": float(c), "idIoU": float(i_),
                          "inkRatio": float(ir), "cosGen": float(cg), "resAlign": float(ra)}
                         for t, b, c, i_, ir, cg, ra in rows]}
    with open("assets/skelnet_amp_calib.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print("[5] 已写 assets/skelnet_amp_calib.json")


if __name__ == "__main__":
    main()
