# -*- coding: utf-8 -*-
"""
eval_diversity.py — 生成多样性检查 (指标 + 视觉海报).

做法: 对同一条件 (书家, 字) 用 **K 个不同初始噪声** 各生成一张, 然后:
  ① 生成之间两两比较 -> 多样性
  ② 生成与 GT 比较    -> 保真度

指标:
  div_mask_iou = 1 - mean(两两笔画 mask IoU)   越高越多样
  div_ssim     = 1 - mean(两两 SSIM)           越高越多样
  ssim_to_gt   = mean(各生成 vs GT 的 SSIM)    越高越准

怎么读 (关键在两者的关系):
  div 高 + ssim_to_gt 低  -> 生成在乱飘 (条件没约束住, 不好)
  div 低 + ssim_to_gt 高  -> 塌缩到单一模式 (多样性不足, 记忆化/过拟合特征)
  div 中 + ssim_to_gt 高  -> 理想: 保真的同时有合理变化

产物:
  assets/diversity_<tag>.csv          每个条件的指标
  assets/diversity_poster_<tag>.png   行=条件 列=K 个生成 + GT (视觉核对)

用法 (需 GPU):
  /opt/conda/envs/cu121/bin/python tools/eval_diversity.py \
      --ckpt <path.pt> --n-cond 8 --k 4 --out-tag sp2base
显存: 训练在跑时可用 --dit-batch 2 挤进去, 或先 SIGSTOP 训练。
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch as th
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

from scipy.ndimage import uniform_filter             # noqa: E402
try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None


def log(m):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ── 指标 ────────────────────────────────────────────────────────────────────
def ssim_np(a, b, win=7):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for ch in range(3):
        x = a[:, :, ch].astype(np.float64)
        y = b[:, :, ch].astype(np.float64)
        mx, my = uniform_filter(x, win), uniform_filter(y, win)
        mx2, my2, mxy = mx ** 2, my ** 2, mx * my
        sx2 = uniform_filter(x * x, win) - mx2
        sy2 = uniform_filter(y * y, win) - my2
        sxy = uniform_filter(x * y, win) - mxy
        out.append((((2 * mxy + c1) * (2 * sxy + c2))
                    / ((mx2 + my2 + c1) * (sx2 + sy2 + c2))).mean())
    return float(np.mean(out))


def mask_iou(a, b, t=0.5):
    b1, b2 = a.mean(2) < t, b.mean(2) < t
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    return float((b1 & b2).sum()) / float((b1 | b2).sum())


# ── 采样 (GPU Heun, 与 cpu_sampler 数学等价) ─────────────────────────────────
@th.no_grad()
def heun_gpu(model, noise, conds, cfg, steps, shift, dev, g=None):
    n = noise.shape[0]
    x = noise.to(dev).float()
    s = th.linspace(1.0, 0.0, steps + 1, dtype=th.float64)
    ts = (shift * s / (1.0 + (shift - 1.0) * s)).tolist() if shift != 1.0 else s.tolist()
    mk = dict(y_callig=th.tensor([c[0] for c in conds], device=dev, dtype=th.long),
              y_char=th.tensor([c[1] for c in conds], device=dev, dtype=th.long))
    if g is not None and getattr(model, "use_glyph_cond", False):
        mk["g"] = g.to(dev).float()
    for k in range(steps):
        t_i, t_nx = ts[k], ts[k + 1]
        dt = t_nx - t_i
        v1 = model.forward_with_cfg(x, th.full((n,), t_i, device=dev) * 1000.0,
                                    cfg_scale=cfg, **mk)
        if isinstance(v1, tuple):
            v1 = v1[0]
        x_e = x + dt * v1
        v2 = model.forward_with_cfg(x_e, th.full((n,), t_nx, device=dev) * 1000.0,
                                    cfg_scale=cfg, **mk)
        if isinstance(v2, tuple):
            v2 = v2[0]
        x = x + dt * 0.5 * (v1 + v2)
    return x.float().cpu()


_G = None   # 当前条件的 g (std skel latent), 由 main 设置


# ckpt args 里数值字段常出现 "key 存在但值 None" (int(None) 会炸, 且默认值不生效),
# 统一清洗后再构建模型。
_ARG_DEFAULTS = {
    "num_calligraphers": 1013, "num_characters": 35130,
    "callig_embed_dim": 128, "char_embed_dim": 384,
    "rope_theta": 100.0, "glyph_scale_init": 0.4,
    "glyph_embedder_depth": 0, "glyph_inject_layers": 0,
    "callig_n_style": 8, "style_token_n": 0, "style_role_init": 0.02,
    "shift": 1.0, "vae_scaling_factor": 0.18215, "latent_channels": 4,
}


def clean_args(a):
    out = dict(a)
    for k, d in _ARG_DEFAULTS.items():
        if out.get(k) is None:
            out[k] = d
    return out


def build_model(a, device, in_channels=4):
    from src.model import DiT_2Cond_models
    arch = dict(norm_type=a.get("norm_type", "rms"),
                mlp_type=a.get("mlp_type", "swiglu"),
                qk_norm=bool(a.get("qk_norm", 1)),
                rope=bool(a.get("rope", 1)),
                rope_theta=float(a.get("rope_theta", 100.0)),
                attn_impl=a.get("attn_impl", "sdpa"))
    m = DiT_2Cond_models[a["model"]](
        num_calligraphers=int(a.get("num_calligraphers", 1013)),
        num_characters=int(a.get("num_characters", 35130)),
        condition_fusion=a.get("condition_fusion", "factorized_add"),
        callig_embed_dim=int(a.get("callig_embed_dim", 128)),
        char_embed_dim=int(a.get("char_embed_dim", 384)),
        char_proj_mode=a.get("char_proj_mode", "mlp"),
        freeze_char_table=bool(a.get("freeze_char_table", False)),
        cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=0.5,
        use_checkpoint=False, learn_sigma=False,
        use_glyph_cond=bool(a.get("skel_as_glyph_cond") or a.get("w_glyph_cond")),
        use_char_cond=not bool(a.get("no_char_cond", False)),
        glyph_scale_init=float(a.get("glyph_scale_init", 0.4)),
        glyph_drop_prob=0.0,
        glyph_embedder_depth=int(a.get("glyph_embedder_depth", 0)),
        glyph_inject_layers=int(a.get("glyph_inject_layers", 0)),
        callig_style_attn=bool(a.get("callig_style_attn", False)),
        callig_n_style=int(a.get("callig_n_style", 8)),
        style_token_n=int(a.get("style_token_n", 0)),
        style_role_init=float(a.get("style_role_init", 0.02)),
        glyph_inject_mode=a.get("glyph_inject_mode", "adaln"),
        input_size=int(a.get("input_size") or 32), in_channels=in_channels,
        **arch)
    if a.get("freeze_callig_table"):
        m.y_callig_embedder.freeze_table()
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-cond", type=int, default=8, help="取多少个条件")
    ap.add_argument("--k", type=int, default=4, help="每条件生成几张 (不同噪声)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--out-tag", default="run")
    ap.add_argument("--eval-csv", default="")
    ap.add_argument("--img-root", default="",
                    help="GT 图根目录; 留空 = 直接用 csv 的 image_path "
                         "(ckpt 里的 img_root 常与 image_path 重复拼接, 导致路径错误)")
    ap.add_argument("--skel-shards", default="",
                    help="骨架 latent shards 目录(覆盖 ckpt args)。迁移后多位于 "
                         "data/skel/ 下, ckpt args 里的裸名字会加载不到 -> g=ZERO")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu = 不占 GPU(不影响训练, ~34s/张); cuda = 快但需显存")
    ap.add_argument("--batch", type=int, default=8, help="CPU 采样 batch")
    ap.add_argument("--threads", type=int, default=16,
                    help="CPU 采样线程数(留余量给训练的数据加载)")
    args = ap.parse_args()

    dev = args.device
    if dev == "cpu":
        th.set_num_threads(args.threads)      # 不抢满 CPU, 避免影响训练吞吐
    global _G
    log(f"ckpt={args.ckpt}")
    ck = th.load(args.ckpt, map_location="cpu", weights_only=False)
    na = ck.get("args", {})
    a = vars(na) if isinstance(na, argparse.Namespace) else (na or {})
    a = clean_args(a)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in (ck.get("ema") or ck.get("model") or ck).items()}
    # v11 是 12ch 模型 (img4 + canny4 + skel4) —— 输入通道必须从权重推断,
    # 用默认 4 会让 x_embedder/final_layer 形状不匹配.
    in_ch = int(sd["x_embedder.proj.weight"].shape[1])
    log(f"in_channels inferred from ckpt = {in_ch}")
    model = build_model(a, dev, in_channels=in_ch)
    miss, unexp = model.load_state_dict(sd, strict=False)
    log(f"load: missing={len(miss)} unexpected={len(unexp)}")
    del ck

    from src.eval.inference import make_eval_cache, load_eval_vae
    from src.utils.callig_map import load_callig_id_map
    csvp = args.eval_csv or (a.get("eval_csv") or "assets/eval_seen_v10.csv")
    # ckpt args 里的 callig_id_map 常是迁移前的旧路径 (5script/...), 逐个候选回退
    cmap = None
    p = a.get("callig_id_map") or ""
    if p:
        cands = [p, os.path.join(ROOT, p),
                 os.path.join(ROOT, "assets", os.path.basename(p)),
                 os.path.join("assets", os.path.basename(p))]
        for c in cands:
            if c and os.path.exists(c):
                cmap, _ = load_callig_id_map(c)
                log(f"callig_id_map: {c}")
                break
        if cmap is None:
            log(f"WARN: callig_id_map not found, tried={cands}")
    n_cond = args.n_cond
    img_root = args.img_root or None
    # shards 目录迁移后多位于 data/skel/ 下; ckpt args 里的裸名字需要补前缀,
    # 否则 8/8 missing skel -> g=ZERO, 字条件完全失效 (测出来的多样性毫无意义).
    shards = args.skel_shards or a.get("skel_latent_shards_dir") or ""
    if shards and not os.path.isdir(shards):
        alt = os.path.join("data/skel", os.path.basename(shards))
        if os.path.isdir(alt):
            log(f"skel shards: {shards} -> {alt}")
            shards = alt
        else:
            log(f"WARN: skel shards 未找到: {shards} -> g 条件将失效")
    cache = make_eval_cache(csvp, img_root, None, 256, n_cond, 8, 4,
                            float(a.get("vae_scaling_factor", 0.18215)),
                            skel_latent_shards_dir=shards,
                            callig_id_map=cmap)
    # 越界兜底: 书家 id 必须 < num_classes, 否则 embedding 索引越界直接崩
    n_cls = int(model.y_callig_embedder.num_classes)
    conds_all, n_over = [], 0
    for c in cache["conds"]:
        cid = int(c[0])
        if cid < 0 or cid >= n_cls:
            cid = n_cls                       # null 行
            n_over += 1
        conds_all.append((cid, int(c[1])))
    if n_over:
        log(f"WARN: {n_over}/{len(conds_all)} callig id 越界 -> 置 null "
            f"(书家条件对这些样本失效)")
    gts_all = cache["gts"]
    gs_all = cache["skels_latent"]
    n_eff = min(n_cond, gts_all.shape[0])
    log(f"conds={n_eff} k={args.k} steps={args.steps} cfg={args.cfg}")

    vae = load_eval_vae(dev, "data/pretrained/sd-vae-ft-ema")
    sf = float(a.get("vae_scaling_factor", 0.18215))

    rows, poster_rows = [], []
    rng = random.Random(0)
    for i in range(n_eff):
        cond = [conds_all[i]]
        _G = gs_all[i:i + 1].float()
        # K 份不同初始噪声 -> 一次采样 K 张 (CPU 上 batch 采样远快于逐张)
        lc = int(a.get("latent_channels") or 4)
        ls = 256 // 8
        noises = [th.randn(1, lc, ls, ls, generator=th.Generator().manual_seed(1000 + k))
                  for k in range(args.k)]
        noise = th.cat(noises, 0)                          # (K, C, H, W)
        g_stack = _G.repeat(args.k, 1, 1, 1)               # 同条件, 重复 K 次
        if args.device == "cpu":
            from src.eval.cpu_sampler import heun_sample_cpu
            lat = heun_sample_cpu(model, noise, cond * args.k, args.cfg,
                                  batch=args.batch, skel=g_stack,
                                  steps=args.steps, shift=float(a.get("shift", 1.0)),
                                  cond_key="g")
        else:
            lat = heun_gpu(model, noise, cond * args.k, args.cfg, args.steps,
                           float(a.get("shift", 1.0)), dev, g=g_stack)
        # 12ch 模型输出 (img4 + canny4 + skel4); VAE 只吃前 latent_channels 通道
        lat_img = lat[:, :lc]
        dec = vae.decode((lat_img.float() / sf)).sample.float().cpu()
        imgs = [((d.clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0) for d in dec]
        gt = ((gts_all[i].clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)

        # 两两多样性
        ious, ssims = [], []
        for x in range(args.k):
            for y in range(x + 1, args.k):
                ious.append(mask_iou(imgs[x], imgs[y]))
                ssims.append(ssim_np(imgs[x], imgs[y]))
        div_iou = 1.0 - float(np.mean(ious)) if ious else 0.0
        div_ssim = 1.0 - float(np.mean(ssims)) if ssims else 0.0
        to_gt = float(np.mean([ssim_np(im, gt) for im in imgs]))
        rows.append({"cond_idx": i, "callig": cond[0][0], "char": cond[0][1],
                     "div_mask_iou": round(div_iou, 4),
                     "div_ssim": round(div_ssim, 4),
                     "ssim_to_gt": round(to_gt, 4)})
        log(f"  cond{i}: div_iou={div_iou:.4f} div_ssim={div_ssim:.4f} "
            f"ssim_to_gt={to_gt:.4f}")
        poster_rows.append(imgs + [gt])

    tag = args.out_tag
    cpath = f"assets/diversity_{tag}.csv"
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 汇总
    summ = {k2: round(float(np.mean([r[k2] for r in rows])), 4)
            for k2 in ("div_mask_iou", "div_ssim", "ssim_to_gt")}
    log(f"MEAN: {summ}")
    with open(cpath.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "k": args.k, "cfg": args.cfg,
                   "n_cond": n_eff, "mean": summ}, f, ensure_ascii=False, indent=2)

    # 海报: 行=条件, 列=K 个生成 + GT
    h, w = poster_rows[0][0].shape[:2]
    canvas = np.ones((h * len(poster_rows), w * (args.k + 1), 3), dtype=np.uint8) * 255
    for r, ims in enumerate(poster_rows):
        for c, im in enumerate(ims):
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = (im * 255).clip(0, 255)
    pp = f"assets/diversity_poster_{tag}.png"
    Image.fromarray(canvas).save(pp)
    log(f"written {cpath} + {pp}  (每行: K个生成 | 最后一列=GT)")


if __name__ == "__main__":
    main()
