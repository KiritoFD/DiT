# -*- coding: utf-8 -*-
"""audit_clean_v1_final.py — 交付前综合抽检.

一次验证所有环节:
  1) 全库质量统计: 图 ink 分布 / 空白数 / 大黑块数 / 标准字 ink 分布
  2) 四联抽检图: [原图 | 标准字 | decode(img_latent) | decode(std_latent)]
     - 左二列应"同一个字", 右二列应与左二列一致 => 配对与 encode 都对
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SF = 0.18215


def ink_of(p):
    try:
        a = np.asarray(Image.open(p).convert("L"))
        return float((a < 128).mean())
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_clean_v1_final.csv")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="/root/Workspace/xy/DiT/_otout_final")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    print(f"[audit] {a.csv}: {len(rows)} 行")

    # ---- 1) 全库质量统计 ----
    import multiprocessing as mp
    with mp.Pool(40) as pool:
        ink_img = pool.map(ink_of, [r["image_path"] for r in rows], chunksize=256)
        ink_std = pool.map(ink_of, [r["std_path"] for r in rows], chunksize=256)
    ink_img, ink_std = np.array(ink_img), np.array(ink_std)
    print(f"\n  [图]   ink: p1={np.percentile(ink_img,1):.4f} "
          f"p50={np.percentile(ink_img,50):.4f} p99={np.percentile(ink_img,99):.4f}")
    print(f"         blank(<0.01)={int((ink_img<0.01).sum())}  "
          f"过重(>0.60)={int((ink_img>0.60).sum())}")
    print(f"  [标准字] ink: p1={np.percentile(ink_std,1):.4f} "
          f"p50={np.percentile(ink_std,50):.4f} p99={np.percentile(ink_std,99):.4f}")
    print(f"         blank(<0.01)={int((ink_std<0.01).sum())}")

    # ---- 2) 四联抽检 ----
    id2loc = {}
    for tag, d in (("img", "data/clean_v1/shards_img"),
                   ("std", "data/clean_v1/shards_std")):
        for sp in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            with np.load(sp) as z:
                for j, iid in enumerate(z["img_ids"]):
                    id2loc[(tag, int(iid))] = (sp, int(j))
    missing = [k for k in id2loc]
    print(f"\n  shard 索引: img+std 共 {len(missing)} 条")

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "data/pretrained/pretrained_models/sd-vae-ft-ema").to("cuda").eval()

    step = max(1, len(rows) // a.n)
    picks = rows[::step][:a.n]
    CELL = 150
    cv = Image.new("RGB", (CELL * 4, CELL * len(picks)), (255, 255, 255))
    for i, r in enumerate(picks):
        iid = int(os.path.basename(r["image_path"])[:-4])
        try:
            im = Image.open(r["image_path"]).convert("RGB").resize((CELL, CELL))
            st = Image.open(r["std_path"]).convert("RGB").resize((CELL, CELL))
        except Exception:
            continue
        dec = {}
        for tag in ("img", "std"):
            if (tag, iid) not in id2loc:
                dec[tag] = Image.new("RGB", (CELL, CELL), (255, 0, 0))
                continue
            sp, j = id2loc[(tag, iid)]
            with np.load(sp) as z:
                lat = np.array(z["latents"][j], dtype=np.float32)
            x = th.from_numpy(lat)[None].to("cuda")
            with th.no_grad():
                o = vae.decode(x / SF).sample
            o = ((o.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            dec[tag] = Image.fromarray((o * 255).astype(np.uint8)).resize((CELL, CELL))
        for k, img in enumerate((im, st, dec["img"], dec["std"])):
            cv.paste(img, (k * CELL, i * CELL))
    cv.save(f"{a.out}/quad.png")
    print(f"  -> {a.out}/quad.png  列: [原图 | 标准字 | decode(img) | decode(std)]")

    # 质量最差的若干 (图 ink 最大/最小) 也导出
    order = np.argsort(ink_img)
    worst = [rows[int(k)] for k in np.concatenate([order[:6], order[-6:]])]
    cv2 = Image.new("RGB", (CELL * 4, CELL * len(worst)), (255, 255, 255))
    for i, r in enumerate(worst):
        iid = int(os.path.basename(r["image_path"])[:-4])
        try:
            im = Image.open(r["image_path"]).convert("RGB").resize((CELL, CELL))
            st = Image.open(r["std_path"]).convert("RGB").resize((CELL, CELL))
        except Exception:
            continue
        cv2.paste(im, (0, i * CELL))
        cv2.paste(st, (CELL, i * CELL))
    cv2.save(f"{a.out}/worst.png")
    print(f"  -> {a.out}/worst.png  (ink 最小6 + 最大6)")


if __name__ == "__main__":
    main()
