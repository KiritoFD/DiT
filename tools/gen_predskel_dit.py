#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_predskel_dit.py — 用 SkelNet-**DiT**(bridge+hide-g 版) 生成评测集的 predskel shards。

与 tools/gen_predskel.py (旧 DeformSkel) 的区别:
  · 模型 = assets/skelnet_dit_G_bridge_nog.pt (DiT_2Cond, rectified flow)
  · 采样 = 50 步 RF, 起点 = 标准骨架 g, **不把 g 喂给网络** (与训练一致, 防泄漏)
  · 幅度校准: z_out = (1-β)*g + β*z_1,  β 取闭式 LS 解的中位数 (实测 0.634)
    —— β* 把 ink 比从 1.59 修到 1.28, 且最小化 ||z_β - x_gt_skel||^2,
       正是下游主干(v26 用 GT 骨架训练)最想要的口径。

产物格式与 GT 骨架 shards 完全一致: npz{latents(N,4,32,32) f16, img_ids(N)},
可直接喂 tools/run_skel_calibration.py 的 --pred-seen/--pred-strict。

用法:
  python tools/gen_predskel_dit.py --set seen20
  python tools/gen_predskel_dit.py --set strict84
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

SETS = {
    "seen20": ("assets/eval_top10_seen_20.csv",
               "data/top10_style23/shards_std",
               "data/top10_style23/predskel_dit_seen20"),
    "strict84": ("assets/eval_top10_strict_subset84.csv",
                 "data/50k_v2_glyph15k/shards_std",
                 "data/top10_style23/predskel_dit_strict84"),
}


def load_shard_index(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        with np.load(f) as z:
            for j, i in enumerate(z["img_ids"]):
                idx[int(i)] = (f, j)
    return idx


class Lazy:
    """缓存整个 latents 数组再按 j 取 (同分片多 id 的致命 bug 已在 v1 修过)"""

    def __init__(self, idx):
        self.idx, self.cur, self.arr = idx, None, None

    def get(self, iid):
        f, j = self.idx[iid]
        if self.cur != f:
            with np.load(f) as z:
                self.arr = z["latents"]
            self.cur = f
        return self.arr[j].astype(np.float32)


def img_id_of(row):
    if row.get("img_id"):
        return int(row["img_id"])
    m = re.search(r"(\d+)\.png$", row["image_path"])
    if not m:
        raise ValueError(f"无法提取 img_id: {row['image_path']}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--resume", default="assets/skelnet_dit_G_bridge_nog.pt")
    ap.add_argument("--beta", type=float, default=0.634,
                    help="幅度校准系数 (tools/calib_skelnet_amp.py 闭式解的中位数)")
    ap.add_argument("--renorm", action="store_true",
                    help="★ 去曝光: decode -> 二值化 -> skeletonize -> 3px膨胀(与GT骨架同配方)\n"
                         "     -> VAE 重编码。把条件强行拉回 v26 的训练分布 (3px GT 骨架),\n"
                         "  治『生成漂白』—— 主干会照抄条件的曝光, predskel 解码越细碎生成越淡。\n"
                         "  实测对照: std骨架(墨3.5xGT)->生成粗壮 ssim0.581; DiT predskel(细碎)\n"
                         "  ->生成漂白 ssim0.640 但 frag 16.6; 旧 DeformSkel(保宽) 0.664/frag5.4。")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=2,
                    help="★ 必须与训练的 --patch 一致 (patch=1 是主推升级)")
    ap.add_argument("--noise-start", action="store_true",
                    help="★ 噪声起步 + **g 当条件** (与主干同构)。训练时**没有**加 --bridge 的\n"
                         "  模型必须用这个。默认(bridge)是 g 起步+hide-g, 两套采样不能混用。\n"
                         "  背景: bridge 起步下 v=0 就等于照抄 g -> 模型被往照抄按 (K 臂实证);\n"
                         "  噪声起步没有这条捷径, 实测 resAlign 更高 (E_inj6 0.5042 / B_w3 0.4933)。")
    ap.add_argument("--inject-layers", type=int, default=2)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--tag", default="",
                    help="输出目录后缀 (如 w7 -> predskel_dit_seen20_w7), 防止覆盖正在被评测读取的 shards")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    csvp, std_dir, out_dir = SETS[a.set]
    if a.tag:
        out_dir = out_dir + "_" + a.tag
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    print(f"[1] {csvp}: {len(rows)} 行; std={std_dir}; β={a.beta}")

    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    from src.utils.callig_script_map import map_callig_script as _mcs

    n_slots = int(csmap.get("num_pairs", 0) or 0)
    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=a.patch, in_channels=4,
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
    st = th.load(a.style_emb, map_location="cpu", weights_only=False)
    emb = st["embedding"] if isinstance(st, dict) else st
    w = model.y_callig_embedder.embedding_table.weight
    if tuple(emb.shape) == (w.shape[0] - 1, w.shape[1]):
        with th.no_grad():
            w[:emb.shape[0]].copy_(emb.float())
    rf = th.load(a.resume, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(dev) for k, v in rf["ema"].items()})
    model.eval()
    print(f"[2] 载入 {a.resume} (step={rf.get('step')})")

    vae = None
    if a.renorm:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
        print("[2a] --renorm: 载入 VAE, 将按 GT 骨架配方 (skeletonize+3px) 重编码")

    sidx = load_shard_index(std_dir)
    lazy = Lazy(sidx)
    print(f"[3] 标准骨架 {len(sidx)} ids")

    os.makedirs(out_dir, exist_ok=True)
    lat, ids, miss = [], [], 0
    shard_i, shard_size = 0, 5000

    def flush():
        nonlocal lat, ids, shard_i
        if not lat:
            return
        p = os.path.join(out_dir, f"shard_{shard_i:05d}.npz")
        np.savez_compressed(p, latents=np.stack(lat).astype(np.float16),
                            img_ids=np.array(ids, dtype=np.int64))
        print(f"    -> {p} ({len(ids)} 条)", flush=True)
        lat, ids = [], []
        shard_i += 1

    B = a.batch
    buf_g, buf_y, buf_id = [], [], []
    with th.no_grad():
        for k, r in enumerate(rows):
            try:
                iid = img_id_of(r)
            except ValueError:
                miss += 1
                continue
            if iid not in sidx:
                miss += 1
                continue
            pid = int(_mcs(int(r["calligrapher_id"]), int(r["script_id"]), csmap))
            buf_g.append(lazy.get(iid))
            buf_y.append(pid)
            buf_id.append(iid)
            if len(buf_g) >= B or k == len(rows) - 1:
                g = th.from_numpy(np.stack(buf_g)).to(dev)
                y = th.tensor(buf_y, dtype=th.long, device=dev)
                # ★ 起点: bridge=g(且不给网络看g) | 噪声起步=纯噪声(把 g 当条件传入)
                z = th.randn_like(g) if a.noise_start else g.clone()
                _gc = g if a.noise_start else th.zeros_like(g)
                ts = th.linspace(1.0, 0.0, a.sample_steps + 1, device=dev)
                for s_ in range(a.sample_steps):
                    out = model(z, th.full((z.shape[0],), float(ts[s_]) * TIME_SCALE,
                                           device=dev),
                                y_callig=y, y_char=th.zeros_like(y), g=_gc)
                    if isinstance(out, tuple):
                        out = out[0]
                    z = z + (ts[s_ + 1] - ts[s_]) * out
                z = (1 - a.beta) * g + a.beta * z                # ★ 幅度校准
                if a.renorm:
                    # ★ 去曝光: 拉回 v26 训练分布 (3px GT 骨架配方)
                    import numpy as _np
                    from skimage.morphology import skeletonize as _sk
                    from scipy.ndimage import binary_dilation as _bd
                    dec = vae.decode(z / 0.18215).sample.mean(1)          # (B,256,256)
                    ink = (dec < 0).cpu().numpy()
                    proc = np.empty_like(ink)
                    for i_ in range(ink.shape[0]):
                        sk = _sk(ink[i_])
                        proc[i_] = _bd(sk, structure=_np.ones((3, 3), bool),
                                       iterations=1)
                    img = th.from_numpy((1.0 - proc.astype(np.float32)) * 2 - 1)
                    img = img.unsqueeze(1).repeat(1, 3, 1, 1).to(dev)     # (B,3,256,256)
                    with th.no_grad():
                        z = vae.encode(img).latent_dist.mode() * 0.18215
                o = z.cpu().numpy()
                for j in range(o.shape[0]):
                    lat.append(o[j])
                    ids.append(buf_id[j])
                buf_g, buf_y, buf_id = [], [], []
                if len(lat) >= shard_size:
                    flush()
    flush()
    tot = 0
    uniq = set()
    for f in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        with np.load(f) as z:
            tot += len(z["img_ids"])
            uniq |= set(int(i) for i in z["img_ids"])
    print(f"[4] DONE -> {out_dir}  共 {tot} 条 / 唯一 {len(uniq)} / 缺骨架 {miss}")


if __name__ == "__main__":
    main()
