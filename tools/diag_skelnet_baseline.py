# -*- coding: utf-8 -*-
"""diag_skelnet_baseline.py — 算 SkelNet 判据 (clDice, tol=3) 的 trivial baseline.

必须与 train_skelnet_dit.py 的判据完全同口径:
  · 预测 = **输入标准骨架本身** (什么都不做, copy baseline)
  · 也给出 "预测 = GT 骨架" 的上界 (应为 ~1.0, 用于自检判据本身没坏)

若 SkelNet 的 clDice <= copy baseline, 说明它并没有学到"该书家的结体",
只是在往平均/空白上靠 —— 即回归目标病态的证据。

用法: python tools/diag_skelnet_baseline.py [--n 128]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _skel(b):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(b)
    except Exception:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        st = generate_binary_structure(2, 2)
        sk, cur = np.zeros_like(b), b.copy()
        while cur.any():
            er = binary_erosion(cur, structure=st)
            sk |= cur & ~er
            cur = er
        return sk


def cldice(pred_b, gt_b, tol=3):
    from scipy.ndimage import binary_dilation
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    sp, sg = _skel(pred_b), _skel(gt_b)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    st = np.ones((3, 3), bool)
    gt_t = binary_dilation(gt_b, structure=st, iterations=tol)
    pr_t = binary_dilation(pred_b, structure=st, iterations=tol)
    tp = float((sp & gt_t).sum()) / float(sp.sum())
    ts = float((sg & pr_t).sum()) / float(sg.sum())
    return 2.0 * tp * ts / (tp + ts) if (tp + ts) > 0 else 0.0


def load_id_map(d):
    mp = {}
    for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(sp) as z:
            for j, i in enumerate(z["img_ids"]):
                mp[int(i)] = (sp, j)
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.val_csv, encoding="utf-8")))
    rows = rows[:a.n]
    cond = load_id_map(a.cond_shards)
    tgt = load_id_map(a.tgt_shards)
    print(f"val {len(rows)} 条; cond ids {len(cond)}, tgt ids {len(tgt)}")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema"
                                        ).to("cuda").eval()

    def get(mp, iid):
        if iid not in mp:
            return None
        sp, j = mp[iid]
        with np.load(sp) as z:
            return th.from_numpy(np.asarray(z["latents"][j])).float()[None].cuda()

    from PIL import Image
    c_copy, c_gt, c_blank, n_ok = [], [], [], 0
    for r in rows:
        iid = int(os.path.basename(r["image_path"]).split(".")[0])
        fp = os.path.join(a.gt_png_dir, f"{iid:06d}.png")
        if not os.path.exists(fp):
            continue
        gt = np.asarray(Image.open(fp).convert("L")) < 128
        g_std = get(cond, iid)
        g_gt = get(tgt, iid)
        if g_std is None:
            continue
        n_ok += 1
        with th.no_grad():
            dec_std = (vae.decode(g_std / 0.18215).sample.mean(1)[0].cpu().numpy() < 0)
        c_copy.append(cldice(dec_std, gt))
        if g_gt is not None:
            with th.no_grad():
                dec_gt = (vae.decode(g_gt / 0.18215).sample.mean(1)[0].cpu().numpy() < 0)
            c_gt.append(cldice(dec_gt, gt))
        c_blank.append(cldice(np.zeros_like(gt), gt))

    def m(x):
        return float(np.mean(x)) if x else float("nan")
    print(f"\n有效 {n_ok} 条")
    print(f"  [上界]  预测=GT 骨架        clDice = {m(c_gt):.4f}   (应接近 1.0, 否则判据/路径有问题)")
    print(f"  ★[baseline] 预测=输入标准骨架  clDice = {m(c_copy):.4f}   <- SkelNet 至少要超过这个")
    print(f"  [下界]  预测=全空白          clDice = {m(c_blank):.4f}")


if __name__ == "__main__":
    main()
