# -*- coding: utf-8 -*-
"""diag_skelnet_steps_sweep.py — 分辨 SkelNet-DiT "没学到" 还是 "采样步数不够"。

对同一 ckpt, 扫 sampling steps ∈ {5,10,20,50,100,200}, 每次算:
  · latent 域 cos(生成, GT)   —— 与 copy baseline cos(输入std,GT) 比较
  · 墨量比 ink(生成)/ink(GT)  —— 看是否随步数收敛到 1
  · clDice (tol=3)           —— 与 copy baseline 0.2666 比较
若步数增大后 clDice 越过 copy baseline, 说明模型学到了、只是采样太粗;
若完全不随步数变好, 说明确实没学到 / 条件信息不足。
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIME_SCALE = 1000.0


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


def cldice(p, g, tol=3):
    from scipy.ndimage import binary_dilation
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp, sg = _skel(p), _skel(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    st = np.ones((3, 3), bool)
    gt_t = binary_dilation(g, structure=st, iterations=tol)
    pr_t = binary_dilation(p, structure=st, iterations=tol)
    tp = float((sp & gt_t).sum()) / float(sp.sum())
    ts = float((sg & pr_t).sum()) / float(sg.sum())
    return 2 * tp * ts / (tp + ts) if (tp + ts) > 0 else 0.0


def cos(a, b):
    a, b = a.ravel().float(), b.ravel().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="assets/skelnet_dit_B_w3.pt.step024000")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", default="5,10,20,50,100,200")
    ap.add_argument("--tol", type=int, default=3)
    a = ap.parse_args()

    sd = th.load(a.ckpt, map_location="cpu", weights_only=False)
    A = sd.get("args", {})
    A = dict(A) if not isinstance(A, argparse.Namespace) else vars(A)
    n_slots = int(sd.get("n_slots", 0))

    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=int(A.get("depth", 6)), hidden_size=int(A.get("hidden", 256)),
        num_heads=int(A.get("heads", 4)), num_calligraphers=max(n_slots, 1),
        num_characters=1, use_char_cond=False, use_glyph_cond=True,
        glyph_in_channels=4, glyph_inject_layers=int(A.get("inject_layers", 2)),
        glyph_inject_mode="adaln", glyph_scale_init=0.6, glyph_drop_prob=0.0,
        glyph_embedder_depth=2, condition_fusion="factorized_cat",
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=float(A.get("style_drop", 0.1)),
        cond_drop_one_prob=0.0, learn_sigma=False,
    ).cuda().eval()
    model.load_state_dict({k: v.cuda() for k, v in sd["ema"].items()}, strict=False)

    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("data/pretrained/pretrained_models/sd-vae-ft-ema"
                                        ).cuda().eval()

    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = None
    if os.path.exists(A.get("callig_map", "")):
        csmap = json.load(open(A["callig_map"], encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=A.get("val_csv"), latent_shards_dir=A.get("tgt_shards"),
        img_root="", image_size=256, is_train=False, preload=True,
        load_image=False, skel_latent_shards_dir=A.get("cond_shards"),
        callig_id_map=None, callig_script_map=csmap)
    rows = list(csv.DictReader(open(A.get("val_csv"), encoding="utf-8")))
    import re
    ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
    n = min(a.n, len(ds))
    gt_png_dir = A.get("gt_png_dir", "")

    def dec(lat):
        with th.no_grad():
            d = vae.decode(lat.cuda() / 0.18215).sample.mean(1)
        return (d < 0).cpu().numpy()

    pred_mode = A.get("pred", "v")
    print(f"ckpt step={sd.get('step')} pred={pred_mode} n={n}\n")

    # copy baseline (同一批样本, 与步数无关)
    cs, cb_ink, cb_cl = [], [], []
    for i in range(n):
        b = ds[i]
        x0 = b["latent"]
        g = b["skel_latent"]
        cs.append(cos(g, x0))
        pg = dec(g[None])[0]
        pg_gt = dec(x0[None])[0]
        cb_ink.append(float(pg.mean() / max(pg_gt.mean(), 1e-9)))
        fp = os.path.join(gt_png_dir, f"{ids[i]:06d}.png")
        if os.path.exists(fp):
            gtm = np.asarray(Image.open(fp).convert("L")) < 128
            cb_cl.append(cldice(pg, gtm, a.tol))
    m = lambda v: float(np.mean(v)) if v else float("nan")
    print(f"[copy baseline] cos={m(cs):.4f}  ink比={m(cb_ink):.3f}  "
          f"clDice={m(cb_cl):.4f}\n")

    print(f"{'steps':>6} {'cos(gen,GT)':>12} {'ink比':>8} {'clDice':>8}")
    print("-" * 40)
    for st in [int(x) for x in a.steps.split(",")]:
        cg, ik, cl = [], [], []
        for i in range(n):
            b = ds[i]
            x0 = b["latent"].float().cuda()[None]
            g = b["skel_latent"].float().cuda()[None]
            y = th.tensor([int(b["y_callig"])], device="cuda")
            z = th.randn_like(x0)
            eps = z.clone()
            ts = th.linspace(1.0, 0.0, st + 1, device="cuda")
            with th.no_grad():
                for k in range(st):
                    out = model(z, th.full((1,), float(ts[k]) * TIME_SCALE, device="cuda"),
                                y_callig=y, y_char=th.zeros_like(y), g=g)
                    if isinstance(out, tuple):
                        out = out[0]
                    if pred_mode == "x0":
                        z = (1 - ts[k + 1]) * out + ts[k + 1] * eps
                    else:
                        z = z + (ts[k + 1] - ts[k]) * out
            cg.append(cos(z[0], x0[0]))
            pg = dec(z)[0]
            pg_gt = dec(x0)[0]
            ik.append(float(pg.mean() / max(pg_gt.mean(), 1e-9)))
            fp = os.path.join(gt_png_dir, f"{ids[i]:06d}.png")
            if os.path.exists(fp):
                gtm = np.asarray(Image.open(fp).convert("L")) < 128
                cl.append(cldice(pg, gtm, a.tol))
        print(f"{st:6d} {m(cg):12.4f} {m(ik):8.3f} {m(cl):8.4f}")


if __name__ == "__main__":
    main()
