# -*- coding: utf-8 -*-
"""verify_skel_match.py — 验证 g 条件(标准字骨架)是否与样本正确配准.

g 条件是主引导信号 (skel_as_glyph_cond=true, glyph_inject_layers=12) —— 若它
与图像目标不匹配, 模型面对的是一组**互相矛盾**的映射, 典型表现就是输出糊团/
黑块且 loss 降不下去。

做法: 取 csv 若干行 -> 按 img_id 从 skel shard 取 latent -> VAE decode ->
     与同一条 csv 记录的**原图**并排导出, 供肉眼核对"骨架是不是这个字"。
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VAE_PATH = "data/pretrained/pretrained_models/sd-vae-ft-ema"
SF = 0.18215


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_base_sym_clean.csv")
    ap.add_argument("--skel", default="data/skel/std_skel3_latents_base_sym")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--aug", default="all")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.aug != "all":
        want = {"plain": ""}.get(a.aug, a.aug)
        rows = [r for r in rows if (r.get("aug") or "") == want]
    print(f"[verify-skel] {a.csv}: {len(rows)} 行 (aug={a.aug})  skel={a.skel}")

    id2loc = {}
    for sp in sorted(glob.glob(os.path.join(a.skel, "shard_*.npz"))):
        with np.load(sp) as d:
            for j, iid in enumerate(d["img_ids"]):
                id2loc[int(iid)] = (sp, int(j))
    print(f"[verify-skel] skel shard index: {len(id2loc)} ids")

    from diffusers.models import AutoencoderKL
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(dev).eval()

    out = "/root/Workspace/xy/DiT/_otout_skel"
    os.makedirs(out, exist_ok=True)
    step = max(1, len(rows) // a.n)
    picks = [rows[i] for i in range(0, len(rows), step)][:a.n]

    CELL = 200
    cv = Image.new("RGB", (CELL * 2 * len(picks), CELL), (255, 255, 255))
    for k, r in enumerate(picks):
        p = r["image_path"]
        iid = int(os.path.basename(p)[:-4])
        if iid not in id2loc:
            print(f"  [{k}] id={iid} 不在 skel shard 中! {p}")
            continue
        sp, j = id2loc[iid]
        with np.load(sp) as d:
            lat = np.array(d["latents"][j], dtype=np.float32)
        x = th.from_numpy(lat)[None].to(dev)
        with th.no_grad():
            dec = vae.decode(x / SF).sample
        dec = ((dec.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = np.asarray(Image.open(p).convert("RGB").resize((CELL, CELL)),
                         dtype=np.float32) / 255.0
        d_img = (dec * 255).astype(np.uint8)
        d_ink = float((np.asarray(Image.fromarray(d_img).convert("L")) < 128).mean())
        cv.paste(Image.fromarray((img * 255).astype(np.uint8)), (k * CELL * 2, 0))
        cv.paste(Image.fromarray(d_img), (k * CELL * 2 + CELL, 0))
        print(f"  [{k}] id={iid} aug={r.get('aug') or '-':3s} "
              f"char={r.get('character','')} script={r.get('script','')} "
              f"callig={r.get('calligrapher','')} 骨架decode_ink={d_ink:.3f}")
    cv.save(f"{out}/pairs.png")
    print(f"  -> {out}/pairs.png  (每对: 左=原图, 右=该 id 的 g 骨架 decode)")


if __name__ == "__main__":
    main()
