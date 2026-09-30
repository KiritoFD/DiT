#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/gen_gt_skel_eval.py — 为评测集生成 **GT 骨架** (aux_skel3 约定) 的 PNG + latent shards。

背景: strict84 的 GT 图来自 `data/50k/imgs/`，其 id 命名空间与所有现成 aux_skel3
目录**都不重叠**（实测 4 个候选目录命中 0/84），所以评测侧历来只有 std skel。
要在 strict84 上评「GT-skel ControlNet」，必须**现算 GT 骨架**。

配方（逐字照搬 tools/gen_aux_50k.py，即 aux_skel3 的定义）:
    ink  = (gray < 128)
    sk   = skeletonize(ink)
    sk3  = binary_dilation(sk, generate_binary_structure(2,2), iterations=1)
    存成 白底(255)黑线(0) 的 L 图
再走 `src.data.vae_io.encode_csv(transform="gray")` 编码成 4ch/32x32 latent，
shard 的 img_id 取自 PNG 文件名（`<id>.png`）——与 eval csv 的 image_path 同源。

自带校验: `--verify-top10` 会对若干 top10 训练图重算骨架并编码，
与 `data/top10_style23/shards_aux_skel3` 里存的 latent 比 cosine —— 若 ≈1 说明配方一致。

用法:
    python tools/gen_gt_skel_eval.py \
        --csv assets/eval_top10_strict_subset84.csv \
        --out-png data/top10_style23/gt_skel_eval_strict84_png \
        --out-shards data/top10_style23/gt_skel_eval_strict84
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

VAE = "pretrained_models/sd-vae-ft-ema"


def iid_of(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


def skel3_of(img_path):
    """返回白底黑线的 3px 骨架 uint8 图。"""
    a = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    ink = a < 128
    if skeletonize is not None:
        sk = skeletonize(ink)
    else:
        sk = ink
    ST = generate_binary_structure(2, 2)
    sk3 = binary_dilation(sk, ST, iterations=1)
    return np.where(sk3, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-shards", required=True)
    ap.add_argument("--img-root", default="")
    ap.add_argument("--verify-top10", action="store_true",
                    help="对 top10 训练图重算骨架, 与已有 shards_aux_skel3 比 cosine")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(a.out_png, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"[gt-skel] {a.csv}: {len(rows)} 行")

    tmp_csv = "/tmp/_gt_skel_eval.csv"
    n_made = n_have = n_missing = 0
    with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "character", "script", "calligrapher"])
        for r in rows:
            ip = r["image_path"]
            if a.img_root and not os.path.isabs(ip) and not ip.startswith(a.img_root):
                ip = os.path.join(a.img_root, ip)
            if not os.path.exists(ip):
                n_missing += 1
                continue
            iid = iid_of(ip)
            if iid is None:
                n_missing += 1
                continue
            out = f"{a.out_png}/{iid:06d}.png"
            if not os.path.exists(out):
                Image.fromarray(skel3_of(ip), "L").save(out)
                n_made += 1
            else:
                n_have += 1
            w.writerow([out, r.get("character", ""), r.get("script", ""),
                        r.get("calligrapher", "")])
    print(f"[gt-skel] 新生成 {n_made}, 已存在 {n_have}, 缺图 {n_missing} -> {a.out_png}")

    from src.data.vae_io import encode_csv
    have_sh = len([f for f in os.listdir(a.out_shards)
                   if f.endswith(".npz")]) if os.path.isdir(a.out_shards) else 0
    if have_sh == 0:
        encode_csv(tmp_csv, a.out_shards, transform="gray", vae_path=VAE,
                   batch=a.batch, shard_size=5000, workers=a.workers, device="cpu")
    else:
        print(f"[gt-skel] shards 已存在 ({have_sh} 个), 跳过编码")
    tot = sum(len(np.load(os.path.join(a.out_shards, f))["img_ids"])
              for f in os.listdir(a.out_shards) if f.endswith(".npz"))
    print(f"[gt-skel] shards -> {a.out_shards}  {tot} latents")

    # ---- 覆盖率自检 ----
    ids = set()
    for f in os.listdir(a.out_shards):
        if f.endswith(".npz"):
            ids |= {int(i) for i in np.load(os.path.join(a.out_shards, f))["img_ids"]}
    want = {iid_of(r["image_path"]) for r in rows} - {None}
    print(f"[gt-skel] 覆盖 {len(want & ids)}/{len(want)}")

    # ---- 配方校验: 与已有 top10 shards_aux_skel3 比 ----
    if a.verify_top10:
        import glob
        import torch
        ref = {}
        for f in sorted(glob.glob("data/top10_style23/shards_aux_skel3/*.npz")):
            with np.load(f) as z:
                for j, i in enumerate(z["img_ids"]):
                    ref[int(i)] = z["latents"][j].astype(np.float32)
        tr = list(csv.DictReader(open("assets/train_top10_style23.csv", encoding="utf-8")))[:6]
        vpng, vsh = "/tmp/_gtverify_png", "/tmp/_gtverify_sh"
        os.makedirs(vpng, exist_ok=True)
        with open("/tmp/_gtverify.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "character", "script", "calligrapher"])
            for r in tr:
                iid = int(r["img_id"])
                p = f"{vpng}/{iid:06d}.png"
                Image.fromarray(skel3_of(r["image_path"]), "L").save(p)
                w.writerow([p, r["character"], r["script"], r["calligrapher"]])
        if os.path.isdir(vsh):
            import shutil
            shutil.rmtree(vsh)
        encode_csv("/tmp/_gtverify.csv", vsh, transform="gray", vae_path=VAE,
                   batch=8, shard_size=100, workers=2, device="cpu")
        got = {}
        for f in sorted(glob.glob(vsh + "/*.npz")):
            with np.load(f) as z:
                for j, i in enumerate(z["img_ids"]):
                    got[int(i)] = z["latents"][j].astype(np.float32)
        print("[verify] 配方 vs 已有 shards_aux_skel3:")
        for iid in sorted(got):
            if iid not in ref:
                print(f"   id={iid}: 参考里没有, 跳过"); continue
            x, y = got[iid].ravel(), ref[iid].ravel()
            cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
            print(f"   id={iid:<6d} cos={cos:.4f}  {'✅' if cos > 0.99 else '⚠️'}")


if __name__ == "__main__":
    main()
