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
    ap.add_argument("--inter-char", type=int, default=0,
                    help="[2026-09-17] 额外做 inter 条件阶段: 取多少个『同一字多书家』的字。"
                         "0=跳过。测的是换书家时输出动不动 —— 即风格控制是真的还是假的。")
    ap.add_argument("--inter-callig", type=int, default=3, help="每个字取几个书家")
    ap.add_argument("--inter-csv", default="assets/train_fame-kxl-tj-px60.csv",
                    help="inter 阶段用的 csv。**默认用训练集** —— 因为 strict 评测集按设计"
                         "是『每字只配一个书家』(unseen 组合), 天然测不了『换书家动不动』。"
                         "inter 是**可控性**测试(条件是否真的起作用), 不是泛化测试, "
                         "用训练数据正当。")
    ap.add_argument("--skip-intra", action="store_true",
                    help="跳过 intra 阶段(同条件换噪声), 只跑 inter。省一半时间。")
    ap.add_argument("--intra-from", default="",
                    help="--skip-intra 时, 从哪份 diversity_*_summary.json 读 intra 参考值。")
    ap.add_argument("--inter-skel-shards", default="",
                    help="inter 阶段用的骨架 shards。**留空则自动**: 用训练集 csv 时必须配"
                         "训练 shards(data/fame-kxl-tj-px60/shards_std), 用评测 csv 才配"
                         "shards_std_eval —— 配错会 100% miss -> g=ZERO -> 字条件失效, "
                         "测出来的数字毫无意义。")
    ap.add_argument("--inter-fixed-noise", type=int, default=1,
                    help="1(默认)=inter 阶段**所有样本共用同一初始噪声** —— 这是必须的: "
                         "只有固定噪声, 样本间差异才**纯粹来自条件**。0=每个样本换噪声"
                         "(会把噪声方差混进条件效应, 得到虚高的 ratio)。")
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
    # ★ 2026-09-17: 本文件原来的 build_model 漏了 v12 之后新增的一批字段
    #   （glyph_vec_cond / glyph_vec_dim / glyph_embedder_sep / xattn_q_pos ...），
    #   而加载用 strict=False —— 实测 v12 的 cond_fusion.0.weight 是 [256]（callig128 +
    #   glyph_vec128）而重建模型给 [128]，**静默加载了随机 cond_fusion**，
    #   测出来的多样性数字毫无意义。
    #   改为复用 tools/cfg_sweep.build_model_from_args（字段逐一对齐 train.py），
    #   并用 **strict=True** 当护栏 —— 字段再漏一个就直接抛错，不会静默。
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("_cfs", os.path.join(ROOT, "tools", "cfg_sweep.py"))
    _cfs = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_cfs)
    import argparse as _ap
    _ns = _ap.Namespace(**a)
    # 补齐 build_model_from_args 需要的字段（它用 getattr 取，缺了会走默认值）
    for _k, _v in (("image_size", 256), ("vae_downscale", 8), ("latent_channels", 4),
                   ("aux_latent_shards_dirs", ""), ("num_calligraphers", 52),
                   ("num_characters", 7765), ("callig_embed_dim", 128),
                   ("char_embed_dim", 384), ("condition_fusion", "factorized_cat")):
        if getattr(_ns, _k, None) is None:
            setattr(_ns, _k, _v)
    # in_channels 以 ckpt 权重为准（12ch 的 aux 组数可能不在 args 里）
    _n_aux = max(0, (in_ch - int(getattr(_ns, "latent_channels", 4) or 4)) // 4)
    if _n_aux:
        _ns.aux_latent_shards_dirs = ",".join(["x"] * _n_aux)
    model = _cfs.build_model_from_args(_ns, dev)
    # 复刻 train.py 构造后的两步后处理（否则 null_embed 缺失）
    _cep = getattr(_ns, "callig_emb_pretrained", "")
    if _cep:
        if not os.path.isabs(_cep) and not os.path.exists(_cep):
            _cep = os.path.join(ROOT, _cep)
        if os.path.exists(_cep):
            _d = th.load(_cep, map_location="cpu", weights_only=False)
            _emb = _d["embedding"] if isinstance(_d, dict) else _d
            with th.no_grad():
                model.y_callig_embedder.embedding_table.weight[:_emb.shape[0]].copy_(_emb.float())
            del _d
            log(f"callig 预训练表已加载 {tuple(_emb.shape)}")
    if getattr(_ns, "freeze_callig_table", False):
        model.y_callig_embedder.freeze_table()
    # ★ 护栏: strict=True。字段再漏一个就抛错，不会静默用随机权重。
    model.load_state_dict(sd, strict=True)
    model.eval()
    log("load OK (strict=True)")
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
    n_eff = 0 if args.skip_intra else min(n_cond, gts_all.shape[0])
    log(f"conds={n_eff} k={args.k} steps={args.steps} cfg={args.cfg}"
        + ("  (--skip-intra: 跳过 intra 阶段)" if args.skip_intra else ""))

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
            from src.utils.cpu_sampler import heun_sample_cpu
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["cond_idx", "callig", "char", "div_mask_iou",
                            "div_ssim", "ssim_to_gt"])
        w.writeheader()
        w.writerows(rows)
    # 汇总
    summ = ({k2: round(float(np.mean([r[k2] for r in rows])), 4)
             for k2 in ("div_mask_iou", "div_ssim", "ssim_to_gt")}
            if rows else {})
    log(f"MEAN: {summ}")
    with open(cpath.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "k": args.k, "cfg": args.cfg,
                   "n_cond": n_eff, "mean": summ}, f, ensure_ascii=False, indent=2)

    # ── inter 条件阶段 (2026-09-17 新增) ────────────────────────────────────
    # 只测 intra(同条件换噪声) 是不够的: 它高只说明"输出有随机变化", 不说明
    # **条件真的起作用**。若模型忽略书家条件, intra 照样很高, 但换书家时输出不动。
    # 关键指标 ratio_style = inter_callig / intra:
    #   >> 1  风格条件有效
    #   ≈ 1   风格条件被噪声淹没(或根本没起作用) -> **假风格控制**
    if args.inter_char > 0:
        from src.utils.cpu_sampler import heun_sample_cpu
        rows_csv = list(csv.DictReader(open(args.inter_csv, encoding="utf-8")))
        by_char = {}
        for r in rows_csv:
            by_char.setdefault(r["character"], []).append(r)
        # 优先选"书家数最多"的字 —— 这样 inter_callig 能取满
        cand = [(c, rs) for c, rs in by_char.items()
                if len({x["calligrapher"] for x in rs}) >= args.inter_callig]
        cand.sort(key=lambda kv: -len({x["calligrapher"] for x in kv[1]}))
        pick_chars = [c for c, _ in cand[:args.inter_char]]
        log(f"[inter] csv={args.inter_csv}  选中 {len(pick_chars)} 个字 x "
            f"{args.inter_callig} 个书家 (候选 {len(cand)} 个字)")
        if not pick_chars:
            log("  WARN: 该 csv 里没有『同一字 ≥inter_callig 个书家』的字 -> 跳过 inter")
        sel = []
        for ch in pick_chars:
            seen, got = set(), 0
            for r in by_char[ch]:
                if r["calligrapher"] in seen:
                    continue
                seen.add(r["calligrapher"])
                sel.append(r)
                got += 1
                if got >= args.inter_callig:
                    break
        # ⚠ 主 cache 只装了前 n_cond 行, 而 inter 需要的是 CSV 里**任意**选中的行
        #   -> 不能拿全 CSV 行号去索引 gts_all/gs_all（会全部越界, 得到空的 inter_imgs）。
        #   做法: 把选中的行写成一个临时子 CSV, 单独建一份 cache, 用 0..len(sel)-1 索引。
        import tempfile
        _tf = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                          dir="assets", encoding="utf-8", newline="")
        _w = csv.DictWriter(_tf, fieldnames=list(rows_csv[0].keys()))
        _w.writeheader()
        _w.writerows(sel)
        _tf.close()
        # ⚠ shards 必须与 csv 配套: 训练 csv 的 img_id 不在 shards_std_eval 里,
        #   配错会 100% miss -> g=ZERO -> 字条件静默失效(实测踩过)。
        _ishards = args.inter_skel_shards
        if not _ishards:
            _ishards = ("data/fame-kxl-tj-px60/shards_std"
                        if "train" in os.path.basename(args.inter_csv) else shards)
        if not os.path.isdir(_ishards):
            log(f"  WARN: inter skel shards 不存在: {_ishards} -> g 将失效")
        log(f"[inter] skel shards = {_ishards}")
        inter_cache = make_eval_cache(_tf.name, img_root, None, 256, len(sel), 8, 4,
                                      float(a.get("vae_scaling_factor", 0.18215)),
                                      skel_latent_shards_dir=_ishards,
                                      callig_id_map=cmap)
        i_gts = inter_cache["gts"]
        i_gs = inter_cache["skels_latent"]
        _nz = int((i_gs.abs().sum(dim=(1, 2, 3)) == 0).sum()) if i_gs is not None else -1
        log(f"[inter] 子 cache: {len(i_gts)} 行, skel {'有' if i_gs is not None else '无'}"
            f"{'' if _nz < 0 else f', 其中全零(g 失效) {_nz}/{i_gs.shape[0]}'}")
        if i_gs is not None and _nz == i_gs.shape[0]:
            log("  ✗ 全部 g=ZERO -> inter 测的是『无条件』的输出差异, 数字无意义。"
                "请用 --inter-skel-shards 指向与 inter_csv 配套的 shards。")
        lc = int(a.get("latent_channels") or 4)
        ls = 256 // 8
        inter_imgs, inter_meta = [], []
        for i, r in enumerate(sel):
            if i >= i_gts.shape[0]:
                continue
            # ★ 必须走词表映射! CSV 里的 calligrapher_id 是**原始 id**(49~9001),
            #   而模型 num_classes=52 -> 不映射就全部越界 -> 兜底成 null
            #   -> 4 个书家输出像素级相同, inter_callig 假性为 0(实测踩过)。
            cid = int(r["calligrapher_id"])
            if cmap:
                cid = int(cmap.get(cid, cid))
            _cid_raw = cid
            cid = cid if 0 <= cid < n_cls else n_cls
            if _cid_raw >= n_cls:
                log(f"  WARN: callig id {_cid_raw} 越界(>= {n_cls}) -> 置 null, 风格条件失效")
            cond = [(cid, int(r["glyph_id"]))]
            gg = i_gs[i:i + 1].float() if i_gs is not None else None
            # ★ 固定噪声: 所有 inter 样本共用同一个初始噪声 -> 样本间差异**纯粹来自条件**。
            #   若每样本换噪声, 测到的是"噪声方差 + 条件效应", ratio 会虚高(实测踩过)。
            _sd = 4242 if args.inter_fixed_noise else (7000 + i)
            noise = th.randn(1, lc, ls, ls, generator=th.Generator().manual_seed(_sd))
            lat = heun_sample_cpu(model, noise, cond, args.cfg, batch=args.batch,
                                  skel=gg, steps=args.steps,
                                  shift=float(a.get("shift", 1.0)), cond_key="g")
            dec = vae.decode((lat[:, :lc].float() / sf)).sample.float().cpu()
            inter_imgs.append(((dec[0].clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0))
            inter_meta.append((r["character"], r["calligrapher"]))
        # 匹配的 intra 参考: **同一批条件**, 换一个噪声再采一遍。
        # 这样 intra/inter 是同数据同口径, ratio 才可比
        # (用评测集的 intra 去比训练集的 inter 是不同数据, 严格说不可比)。
        intra_imgs = []
        if args.inter_fixed_noise:
            for i, r in enumerate(sel):
                if i >= i_gts.shape[0]:
                    continue
                cid = int(r["calligrapher_id"])
                if cmap:
                    cid = int(cmap.get(cid, cid))
                cid = cid if 0 <= cid < n_cls else n_cls
                gg = i_gs[i:i + 1].float() if i_gs is not None else None
                noise = th.randn(1, lc, ls, ls, generator=th.Generator().manual_seed(9999 + i))
                lat = heun_sample_cpu(model, noise, [(cid, int(r["glyph_id"]))], args.cfg,
                                      batch=args.batch, skel=gg, steps=args.steps,
                                      shift=float(a.get("shift", 1.0)), cond_key="g")
                dec = vae.decode((lat[:, :lc].float() / sf)).sample.float().cpu()
                intra_imgs.append(((dec[0].clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0))
        os.unlink(_tf.name)
        log(f"[inter] 采样完成 {len(inter_imgs)} 张 inter + {len(intra_imgs)} 张匹配 intra")
        inter_c, inter_h = [], []
        # ★ 2026-09-18: 原来只留聚合均值，**逐对数据被丢弃** -> 无法做"按书家拆开"的分析
        #   （例如"该书家的风格条件有多强" vs "该书家的 strict ssim" 的散点，
        #    用来区分瓶颈在『数据』还是『条件机制』）。
        #   现在把逐对 + 逐书家聚合都落盘。
        # ⚠ 元组顺序是 (character, calligrapher)（见 inter_meta 构造处），
        #   **不是** (callig, glyph)。第一次实现时把变量名写反，导致按书家聚合
        #   实际按"字"聚合了（输出里出现 爲/為/東/有…）。下面用具名变量避免再错。
        _pairs = []          # (kind, char_x, cal_x, char_y, cal_y, ssim)
        for x in range(len(inter_imgs)):
            for y in range(x + 1, len(inter_imgs)):
                (chx, cax), (chy, cay) = inter_meta[x], inter_meta[y]
                s = ssim_np(inter_imgs[x], inter_imgs[y])
                if chx == chy and cax != cay:
                    inter_c.append(s)          # 同字换书家
                    _pairs.append(("callig", chx, cax, chy, cay, s))
                if cax == cay and chx != chy:
                    inter_h.append(s)          # 同书家换字
                    _pairs.append(("char", chx, cax, chy, cay, s))
        # intra 参考值: 优先用本轮; --skip-intra 时从 --intra-from 的 summary json 读
        if rows:
            m_intra = float(np.mean([r["div_ssim"] for r in rows]))
        elif intra_imgs and len(intra_imgs) == len(inter_imgs):
            m_intra = float(np.mean([1.0 - ssim_np(a, b)
                                     for a, b in zip(intra_imgs, inter_imgs)]))
            log(f"[inter] intra 取自**匹配采样**(同条件换噪声, 同数据同口径): {m_intra:.4f}")
        elif args.intra_from and os.path.exists(args.intra_from):
            _j = json.load(open(args.intra_from, encoding="utf-8"))
            m_intra = float(_j["mean"].get("div_ssim", 0.0))
            log(f"[inter] intra 取自 {args.intra_from}: div_ssim={m_intra:.4f}")
        else:
            m_intra = 0.0
            log("[inter] WARN: 无 intra 参考值 -> ratio_style 无法计算 "
                "(用 --intra-from 指向上一次的 summary json)")
        d_c = 1.0 - float(np.mean(inter_c)) if inter_c else float("nan")
        d_h = 1.0 - float(np.mean(inter_h)) if inter_h else float("nan")
        summ["div_intra_ssim"] = round(m_intra, 4)
        summ["div_inter_callig"] = round(d_c, 4) if inter_c else None
        summ["div_inter_char"] = round(d_h, 4) if inter_h else None
        summ["ratio_style"] = round(d_c / m_intra, 2) if (inter_c and m_intra > 0) else None
        summ["ratio_char"] = round(d_h / m_intra, 2) if (inter_h and m_intra > 0) else None
        log(f"[inter] intra={m_intra:.4f} inter_callig={d_c:.4f} inter_char={d_h:.4f} "
            f"ratio_style={summ['ratio_style']} ratio_char={summ['ratio_char']}")
        log("  判读: ratio >> 1 条件有效; ratio ≈ 1 条件被噪声淹没(假风格控制)")

        # ── 落盘：逐对 + 逐书家聚合（★ 2026-09-18 新增）──────────────────
        # 用途：把「某书家的风格条件有多强」与「该书的 strict ssim」对齐画散点，
        #       区分瓶颈在『数据不足』还是『条件机制没生效』。
        _pc = f"assets/diversity_inter_pairs_{tag}.csv"
        with open(_pc, "w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow(["kind", "char_a", "callig_a", "char_b", "callig_b",
                         "ssim", "diff"])
            for k, cha, caa, chb, cab, s in _pairs:
                _w.writerow([k, cha, caa, chb, cab, round(float(s), 6),
                             round(1.0 - float(s), 6)])
        log(f"  written {_pc}  ({len(_pairs)} 对)")

        # 逐书家聚合：对每个书家，取"固定字、换书家"里**涉及该书的**那几对的 1-SSIM
        # （该书家作为 a 或 b 都算），再对字求平均。
        from collections import defaultdict as _dd
        _by_cal = _dd(list)
        for k, cha, caa, chb, cab, s in _pairs:
            if k != "callig":
                continue
            _by_cal[caa].append(1.0 - float(s))
            _by_cal[cab].append(1.0 - float(s))
        _cc = f"assets/diversity_inter_by_callig_{tag}.csv"
        with open(_cc, "w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow(["callig", "n_pairs", "mean_diff_style"])
            for c in sorted(_by_cal, key=lambda z: -float(np.mean(_by_cal[z]))):
                v = _by_cal[c]
                _w.writerow([c, len(v), round(float(np.mean(v)), 6)])
        log(f"  written {_cc}  ({len(_by_cal)} 个书家)")

        if not inter_imgs:
            log("  WARN: inter_imgs 为空 -> 跳过 inter 海报")
            h, w = poster_rows[0][0].shape[:2]
            canvas = np.ones((h * len(poster_rows), w * (args.k + 1), 3), dtype=np.uint8) * 255
            for r_, ims in enumerate(poster_rows):
                for c_, im in enumerate(ims):
                    canvas[r_ * h:(r_ + 1) * h, c_ * w:(c_ + 1) * w] = (im * 255).clip(0, 255)
            pp = f"assets/diversity_poster_{tag}.png"
            Image.fromarray(canvas).save(pp)
            with open(cpath.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"ckpt": args.ckpt, "k": args.k, "cfg": args.cfg,
                           "n_cond": n_eff, "mean": summ}, f, ensure_ascii=False, indent=2)
            log(f"written {cpath} + {pp}")
            return
        # 海报
        hh, ww = inter_imgs[0].shape[:2]
        cv2 = np.ones((hh * len(inter_imgs), ww, 3), dtype=np.uint8) * 255
        for rr, im in enumerate(inter_imgs):
            cv2[rr * hh:(rr + 1) * hh, :ww] = (im * 255).clip(0, 255)
        Image.fromarray(cv2).save(f"assets/diversity_inter_{tag}.png")
        log(f"written assets/diversity_inter_{tag}.png")

    # 海报: 行=条件, 列=K 个生成 + GT
    if not poster_rows:
        log("(--skip-intra: 无 intra 海报; summary 已重写)")
        with open(cpath.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "k": args.k, "cfg": args.cfg,
                       "n_cond": n_eff, "mean": summ}, f, ensure_ascii=False, indent=2)
        return
    h, w = poster_rows[0][0].shape[:2]
    canvas = np.ones((h * len(poster_rows), w * (args.k + 1), 3), dtype=np.uint8) * 255
    for r, ims in enumerate(poster_rows):
        for c, im in enumerate(ims):
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = (im * 255).clip(0, 255)
    pp = f"assets/diversity_poster_{tag}.png"
    Image.fromarray(canvas).save(pp)
    log(f"written {cpath} + {pp}  (每行: K个生成 | 最后一列=GT)")
    # inter 阶段会往 summ 里追加指标 -> 重写一次 summary (原 dump 在 inter 之前)
    with open(cpath.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "k": args.k, "cfg": args.cfg,
                   "n_cond": n_eff, "mean": summ}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
