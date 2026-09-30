#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""poster_predskel.py — 可视化 SkelNet-DiT 的骨架形变质量。

每行一个样本, 四列: 标准骨架(输入g) | 模型raw(β=1) | 校准(β=0.634) | GT真迹骨架。
底部再拼一行: 用两种 g 喂冻结 v26 主干的**生成结果**对照 (可选, --with-main-out)。

用法: python tools/poster_predskel.py --n 12
"""
import argparse
import csv
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
FONT_CANDS = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/opt/conda/envs/cu121/fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]


def get_font(sz):
    from PIL import ImageFont
    for p in FONT_CANDS:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w3")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--resume", default="assets/skelnet_dit_G_bridge_nog.pt")
    ap.add_argument("--beta", type=float, default=0.634)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--cell", type=int, default="128" and 128)
    ap.add_argument("--out", default="_ot_scratch/predskel_poster.png")
    a = ap.parse_args()
    dev = "cuda"

    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=a.val_csv, latent_shards_dir=a.tgt_shards, img_root="",
        image_size=256, is_train=False, preload=True, load_image=False,
        skel_latent_shards_dir=a.cond_shards,
        callig_id_map=None, callig_script_map=csmap)
    n_slots = int(csmap.get("num_pairs", 0) or 0)

    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4,
        depth=6, hidden_size=256, num_heads=4,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=2, glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0, glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=0.1, cond_drop_one_prob=0.0, learn_sigma=False,
    ).to(dev)
    st = th.load(a.style_emb, map_location="cpu", weights_only=False)
    emb = st["embedding"] if isinstance(st, dict) else st
    w = model.y_callig_embedder.embedding_table.weight
    with th.no_grad():
        w[:emb.shape[0]].copy_(emb.float())
    rf = th.load(a.resume, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(dev) for k, v in rf["ema"].items()})
    model.eval()

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()

    # 选样: 跳过前 32 (eval_val 用的是它们), 取接下来的 n 个, 避免只看同字
    n = a.n
    sel = list(range(32, 32 + n))
    rows_meta = []
    with th.no_grad():
        for i in sel:
            b = ds[i]
            x0 = b["latent"].float().unsqueeze(0).to(dev)
            g = b["skel_latent"].float().unsqueeze(0).to(dev)
            y = th.tensor([int(b["y_callig"])], device=dev)
            iid = int(b["img_id"])
            z = g.clone()
            ts = th.linspace(1.0, 0.0, a.sample_steps + 1, device=dev)
            for k in range(a.sample_steps):
                out = model(z, th.full((1,), float(ts[k]) * TIME_SCALE, device=dev),
                            y_callig=y, y_char=th.zeros_like(y, device=dev),
                            g=th.zeros_like(g))
                if isinstance(out, tuple):
                    out = out[0]
                z = z + (ts[k + 1] - ts[k]) * out
            d_std = vae.decode(g / 0.18215).sample.mean(1)[0].cpu().numpy()
            d_raw = vae.decode(z / 0.18215).sample.mean(1)[0].cpu().numpy()
            zc = (1 - a.beta) * g + a.beta * z
            d_cal = vae.decode(zc / 0.18215).sample.mean(1)[0].cpu().numpy()
            fp = os.path.join(a.gt_png_dir, f"{iid:06d}.png")
            from PIL import Image
            gt = np.asarray(Image.open(fp).convert("L")) if os.path.exists(fp) \
                else np.full((256, 256), 255, np.uint8)
            rows_meta.append(dict(iid=iid, std=d_std, raw=d_raw, cal=d_cal,
                                  gt=gt, y=int(b["y_callig"])))
        print(f"[gen] {len(rows_meta)} 样本完成")

    # ── 拼图 ──
    from PIL import Image, ImageDraw
    cell, lab = a.cell, 22
    cols = ["std(input)", "raw b=1", f"calib b={a.beta}", "GT"]
    cv = Image.new("L", (4 * cell, len(rows_meta) * (cell + lab)), 255)
    dr = ImageDraw.Draw(cv)
    fnt = get_font(14)
    for c, name in enumerate(cols):
        dr.text((c * cell + 6, 2), name, fill=0, font=fnt)
    for r, m in enumerate(rows_meta):
        yy = r * (cell + lab) + lab
        for c, key in enumerate(["std", "raw", "cal", "gt"]):
            if key == "gt":
                arr = m[key]
            else:
                # VAE 解码值域约 [-1.5,+0.3], 墨=暗=<0; 必须二值化, 否则全黑
                arr = np.where(m[key] < 0, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
            # 骨架图是白底黑墨; 统一 反相成 黑底白墨? 保持白底黑墨 (GT png 同)
            cv.paste(im.resize((cell, cell)), (c * cell, yy))
        dr.text((4 * cell - 60, yy + 4), f"id{m['iid']}", fill=128, font=fnt)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv.save(a.out)
    print(f"[out] {a.out}  ids={[m['iid'] for m in rows_meta]}")


if __name__ == "__main__":
    main()
