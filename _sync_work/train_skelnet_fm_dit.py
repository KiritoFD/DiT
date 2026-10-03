#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_skelnet_fm_dit.py — 64x64 **像素域 Flow Matching**，网络**完全照抄 v25/v26**。

用户 2026-09-30 裁定: 不要手搓 U-Net (无全局注意力, 笔画间无法全局协调),
直接抄仓库自己的主干架构 (v13/v25 谱系)。

照抄项 (对照 src/train/configs/v25_stdskel.json / v26_gtskel.json):
  model          = DiT-2Cond-S/2        (depth 12 / hidden 384 / heads 6, ~33M)
  g 通路          = use_glyph_cond + skel_as_glyph_cond 直通 (glyph_inject_layers=4,
                   glyph_embedder_depth=2, glyph_scale_init=0.6, glyph_drop_prob=0.0)
  风格            = y_callig (pair_id 0..22, 表初始化为 callig_script_emb_top10.pt)
  condition_fusion= factorized_cat,  callig_embed_dim=128
  flow            = linear interpolant, velocity 目标, 纯 MSE, t~logit_normal(0,1)
  sampler         = Heun 二阶 + shift 网格, 时间 ×1000 喂模型

与 v25/v26 的唯一差异 (因为我们要的是**骨架**而不是字图):
  · 空间 = **64x64 像素** (input_size=64, patch=2 -> 32x32 tokens), in/out_channels=1
    (v25/v26 是 32x32 的 4 通道 VAE latent)
  · 额外加了**结构 loss + 单边墨量约束** (像素域稀疏二值目标, 纯 MSE 会被 94.5%
    平凡背景像素主导 —— 实测纯 MSE 下 train loss 掉到 0.14 但采样结构仍等于噪声)

数据: tools/build_skel64_dataset.py 落盘的 64² 数据集 (x0=gt_skel, cond=std_skel)

用法:
  python tools/train_skelnet_fm_dit.py --steps 3000 --batch 256
  python tools/train_skelnet_fm_dit.py --poster 8 --resume <ckpt>
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch as th
import torch.nn.functional as F

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TIME_SCALE = 1000.0


def soft_dice_prob(p, tgt, eps=1.0):
    num = 2 * (p * tgt).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64,
                    help="工作分辨率 (像素域)。32 = 用简单下采样代替 VAE:\n"
                         "  VAE 的空间尺度本来就是 32x32 (256/8), 所以 32² 与主干\n"
                         "  几何尺度天然对齐; token 数 16x16=256, 正好是 S/2 的设计点,\n"
                         "  且每步比 64² 便宜 4 倍。1px@32 ≈ 8px@256 (与效果最好的\n"
                         "  w7@256 载体同量级)")
    ap.add_argument("--dataset", default="data/top10_style23/skel64")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--model", default="DiT-2Cond-S/2")
    ap.add_argument("--glyph-inject-layers", type=int, default=4)
    ap.add_argument("--glyph-inject-mode", default="adaln",
                    choices=["adaln", "xattn"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    # ★ 以 **epoch** 为训练单位 (用户裁定): 1 epoch = 训练集过 passes_per_epoch 遍
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--passes-per-epoch", type=int, default=1,
                    help="1 个 epoch = 把训练集过几遍 (默认 1 遍)")
    ap.add_argument("--eval-every-epochs", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1,
                    help="torch.compile (主干 config 的 compile:true, 之前漏抄)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--es-patience", type=int, default=5)
    ap.add_argument("--t-std", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--sampler", choices=["heun", "euler"], default="heun")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--w-struct", type=float, default=0.5)
    ap.add_argument("--struct-min-t", type=float, default=0.4)
    ap.add_argument("--struct-max-t", type=float, default=1.0)
    ap.add_argument("--w-ink-ratio", type=float, default=0.3)
    ap.add_argument("--ink-ratio-tol", type=float, default=1.5)
    ap.add_argument("--max-ink", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--resume", default="")
    ap.add_argument("--poster", type=int, default=0)
    ap.add_argument("--poster-out", default="_ot_scratch/fm_dit_poster.png")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--dump-width", type=int, default=3)
    ap.add_argument("--dump-tag", default="")
    ap.add_argument("--out", default="assets/skelnet_fm_dit.pt")
    ap.add_argument("--log", default="logs/skelnet_fm_dit.log")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    def log(m):
        print(m, flush=True)

    csmap = json.load(open(a.callig_map, encoding="utf-8"))
    from src.utils.callig_script_map import map_callig_script as _mcs
    st_ = th.load(a.style_emb, map_location="cpu", weights_only=False)
    style_tab = (st_["embedding"] if isinstance(st_, dict) else st_).float()

    n_slots = int(csmap.get("num_pairs", 0) or 0)
    from src.model.dit import DiT_2Cond_models
    model = DiT_2Cond_models[a.model](
        # ⚠ 注册表函数 (DiT_2Cond_S_2 等) 内部已固定 depth/hidden/heads/**patch_size**,
        #   这里再传 patch_size 会 "got multiple values" 直接崩
        input_size=a.res, in_channels=1, out_channels=1,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=1,
        glyph_inject_layers=a.glyph_inject_layers,
        glyph_inject_mode=a.glyph_inject_mode,
        glyph_scale_init=0.6, glyph_drop_prob=0.0, glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=a.cond_drop, cond_drop_one_prob=0.0,
        learn_sigma=False).to(dev)
    w = model.y_callig_embedder.embedding_table.weight
    if tuple(style_tab.shape) == (w.shape[0] - 1, w.shape[1]):
        with th.no_grad():
            w[:style_tab.shape[0]].copy_(style_tab.to(dev))
        log(f"[0] 载入预训练风格表 {tuple(style_tab.shape)}")
    log(f"[0] {a.model} (照抄 v25/v26): "
        f"{sum(p.numel() for p in model.parameters()):,} 参数, "
        f"g 通路 inject_layers={a.glyph_inject_layers} mode={a.glyph_inject_mode}")

    # ── 数据 ─────────────────────────────────────────────────────────────
    def load_ds(split):
        p = os.path.join(a.dataset, f"{split}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"[FATAL] 缺落盘数据集 {p}")
        z = np.load(p)
        ids = [int(i) for i in z["ids"]]
        return ({i: k for k, i in enumerate(ids)}, z["std_skel"], z["gt_skel"],
                z["img_ink"])

    def rows_with_y(csvp):
        out = []
        for r in csv.DictReader(open(csvp, encoding="utf-8")):
            try:
                i = int(r["img_id"])
            except Exception:                                  # noqa: BLE001
                m = re.search(r"(\d+)\.png$", r["image_path"])
                if not m:
                    continue
                i = int(m.group(1))
            out.append((i, int(_mcs(int(r["calligrapher_id"]),
                                    int(r["script_id"]), csmap))))
        return out

    tr_mp, tr_std, tr_gt, tr_img = load_ds("train")
    va_mp, va_std, va_gt, va_img = load_ds("val")
    _sc0 = 255.0 if tr_img.max() > 1.5 else 1.0
    _ok_tr = set(int(i) for i in np.array(list(tr_mp))[
        tr_img.reshape(len(tr_img), -1).mean(1) / _sc0 <= a.max_ink])
    _ok_va = set(int(i) for i in np.array(list(va_mp))[
        va_img.reshape(len(va_img), -1).mean(1) / _sc0 <= a.max_ink])
    tr_rows = [r for r in rows_with_y("assets/train_top10_style23_minusval.csv")
               if r[0] in tr_mp and r[0] in _ok_tr]
    va_rows = [r for r in rows_with_y("assets/val_skelnet.csv")
               if r[0] in va_mp and r[0] in _ok_va]
    TR_G = th.from_numpy(tr_std[[tr_mp[r[0]] for r in tr_rows]].astype(np.float32))
    TR_T = th.from_numpy(tr_gt[[tr_mp[r[0]] for r in tr_rows]].astype(np.float32))
    TR_Y = np.array([r[1] for r in tr_rows], np.int64)
    n_va = min(a.val_n, len(va_rows))
    va_ids = [r[0] for r in va_rows[:n_va]]
    VA_G = th.from_numpy(va_std[[va_mp[i] for i in va_ids]].astype(np.float32))
    VA_T = th.from_numpy(va_gt[[va_mp[i] for i in va_ids]].astype(np.float32))
    VA_Y = np.array([r[1] for r in va_rows[:n_va]], np.int64)
    _sc = 255.0 if TR_T.max() > 1.5 else 1.0
    log(f"[1] 数据: 训练 {len(tr_rows)} 验证 {n_va} | 量纲 0/{int(_sc)} | "
        f"x0 墨比 {float(TR_T.mean())/_sc:.4f} | g 墨比 {float(TR_G.mean())/_sc:.4f}")

    def to_x(v):
        v = th.as_tensor(v).float()
        if float(v.max()) > 1.5:
            v = v / 255.0
        return 1.0 - 2.0 * v                      # 墨=-1, 背景=+1

    # ── epoch 定义 ───────────────────────────────────────────────────────
    steps_per_epoch = (max(1, int(math.ceil(len(tr_rows) / a.batch)))
                       * max(1, a.passes_per_epoch))
    total_steps = a.epochs * steps_per_epoch
    eval_every = steps_per_epoch * max(1, a.eval_every_epochs)
    log(f"[1a] 1 epoch = {steps_per_epoch} 步 (= {len(tr_rows)}/{a.batch} × "
        f"{a.passes_per_epoch} 遍); 共 {a.epochs} epochs = {total_steps} 步; "
        f"每 {a.eval_every_epochs} epoch 评测一次 (每 {eval_every} 步)")

    raw = model              # 原始模块; 评测/poster/dump 一律用它 (eager)
    opt = th.optim.AdamW(raw.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1), 1.0) *
        (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(total_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in raw.state_dict().items()}

    def sample_t(n):
        return th.sigmoid(th.randn(n, device=dev) * a.t_std)

    def vel(m, x, t, g, y):
        """m: 训练用 compiled 模块 (快), 评测用 raw 原始模块 (避免 load_state_dict
        触发 torch.compile 重新 trace)。两者共享同一组 Parameters。"""
        return m(x, t * TIME_SCALE, y_callig=y, y_char=th.zeros_like(y), g=g)

    @th.no_grad()
    def heun(g, y, steps):
        b = g.shape[0]
        x = th.randn_like(g)
        ss = np.linspace(1.0, 0.0, steps + 1)
        if a.shift != 1.0:
            ss = a.shift * ss / (1.0 + (a.shift - 1.0) * ss)
        for k in range(steps):
            t_i, t_n = float(ss[k]), float(ss[k + 1])
            dt = t_n - t_i
            with th.autocast("cuda", dtype=th.bfloat16):     # ★ AMP: 实测 fp32 只有
                v1 = vel(raw, x, th.full((b,), t_i, device=dev), g, y)   # 0.86 TFLOPS
            v1 = v1.float()
            if a.sampler == "heun":
                with th.autocast("cuda", dtype=th.bfloat16):
                    v2 = vel(raw, x + dt * v1, th.full((b,), t_n, device=dev), g, y)
                v2 = v2.float()
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x + dt * v1
        return x

    def predict(g_u8, ys, use_ema=True):
        raw.eval()
        ck = None
        if use_ema:
            ck = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            raw.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        out = []
        with th.no_grad():
            for s0 in range(0, len(g_u8), 64):
                gg = to_x(g_u8[s0:s0 + 64]).unsqueeze(1).to(dev)
                yy = th.tensor(ys[s0:s0 + 64], device=dev)
                out.append((heun(gg, yy, a.sample_steps) < 0).float()
                           [:, 0].cpu().numpy())
        if ck is not None:
            raw.load_state_dict(ck)
        raw.train()
        return np.concatenate(out).astype(np.uint8)

    from scipy.ndimage import binary_dilation as _bd

    def eval_val():
        P = predict(VA_G, VA_Y)
        tt = (VA_T.numpy() > 127)
        st8 = np.ones((3, 3), bool)
        dice, prec, rec = [], [], []
        for i in range(P.shape[0]):
            pi, ti = P[i] > 0, tt[i]
            dice.append(2.0 * (pi & ti).sum() / max(int(pi.sum() + ti.sum()), 1))
            prec.append(float((pi & _bd(ti, st8, 1)).sum())
                        / max(int(pi.sum()), 1))
            rec.append(float((ti & _bd(pi, st8, 1)).sum())
                       / max(int(ti.sum()), 1))
        return {"dice64": float(np.mean(dice)), "prec64": float(np.mean(prec)),
                "rec64": float(np.mean(rec)),
                "lostInk": float(1.0 - np.mean(rec)),
                "inkRatio": float(P.mean() / max(tt.mean(), 1e-9))}

    # ── poster ───────────────────────────────────────────────────────────
    if a.poster > 0:
        rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v.float() for k, v in rf["ema"].items()})
        n = max(1, min(a.poster, n_va))
        idxs = np.linspace(0, n_va - 1, n).astype(int)
        Gsel, Tsel, Ysel = VA_G[idxs], VA_T[idxs], VA_Y[idxs]
        P = predict(Gsel, Ysel)
        tt = (Tsel.numpy() > 127)
        gg = (Gsel.numpy() > 127)
        st8 = np.ones((3, 3), bool)
        log(f"[poster] ckpt={a.resume} step={rf.get('step')} n={n} "
            f"steps={a.sample_steps}")
        log(f"  {'#':>3} {'id':>8} {'采样墨比':>9} {'目标墨比':>9} {'墨量倍':>7} "
            f"{'Dice':>7} {'rec':>7} {'prec':>7}")
        d_l, r_l, p_l, m_l = [], [], [], []
        for r in range(n):
            pi, ti = P[r] > 0, tt[r]
            d = 2.0 * (pi & ti).sum() / max(int(pi.sum() + ti.sum()), 1)
            rc = float((ti & _bd(pi, st8, 1)).sum()) / max(int(ti.sum()), 1)
            pr = float((pi & _bd(ti, st8, 1)).sum()) / max(int(pi.sum()), 1)
            gm = float(pi.mean()) / max(float(ti.mean()), 1e-9)
            d_l.append(d); r_l.append(rc); p_l.append(pr); m_l.append(gm)
            log(f"  {r:>3} {va_ids[idxs[r]]:>8} {pi.mean():9.4f} {ti.mean():9.4f} "
                f"{gm:7.2f} {d:7.4f} {rc:7.4f} {pr:7.4f}")
        log(f"  均值: 墨量倍 {np.mean(m_l):.2f} Dice {np.mean(d_l):.4f} "
            f"rec {np.mean(r_l):.4f} prec {np.mean(p_l):.4f}")
        from PIL import Image, ImageDraw, ImageFont
        cell = 130
        cv = Image.new("L", (3 * cell, n * (cell + 20)), 255)
        dr = ImageDraw.Draw(cv)
        try:
            fnt = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:                                      # noqa: BLE001
            fnt = ImageFont.load_default()
        for c, nm in enumerate(["cond g", "采样", "目标 x0"]):
            dr.text((c * cell + 4, 3), nm, fill=0, font=fnt)
        for r in range(n):
            yy = r * (cell + 20) + 18
            for c, arr in enumerate([gg[r], P[r] > 0.5, tt[r]]):
                im = Image.fromarray(np.where(arr, 0, 255).astype(np.uint8))
                cv.paste(im.resize((cell, cell), Image.NEAREST), (c * cell, yy))
        os.makedirs(os.path.dirname(a.poster_out) or ".", exist_ok=True)
        cv.save(a.poster_out)
        log(f"[poster] -> {a.poster_out}")
        return

    # ── dump ─────────────────────────────────────────────────────────────
    if a.dump:
        rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v.float() for k, v in rf["ema"].items()})
        from skimage.morphology import skeletonize as _sk
        from diffusers.models import AutoencoderKL
        for name in ("seen20", "strict84"):
            mp_, std_sk, _gt, _ink = load_ds(name)
            csvp = ("assets/eval_top10_seen_20.csv" if name == "seen20"
                    else "assets/eval_top10_strict_subset84.csv")
            rws = [r for r in rows_with_y(csvp) if r[0] in mp_]
            ids = [r[0] for r in rws]
            ys = [r[1] for r in rws]
            p64 = predict(std_sk[[mp_[i] for i in ids]], ys)
            up = F.interpolate(th.from_numpy(p64).float()[:, None],
                               size=(256, 256), mode="nearest")[:, 0].numpy() > 0.5
            _it = max(0, (a.dump_width - 1) // 2)
            proc = np.empty_like(up)
            for i in range(up.shape[0]):
                sk = _sk(up[i])
                proc[i] = (_bd(sk, structure=np.ones((3, 3), bool),
                               iterations=_it) if _it > 0 else sk)
            vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
            img = th.from_numpy((1.0 - proc.astype(np.float32)) * 2 - 1)
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
            zl = []
            for s0 in range(0, img.shape[0], 8):
                with th.no_grad(), th.autocast("cuda", dtype=th.float16):
                    zl.append(vae.encode(img[s0:s0 + 8].to(dev)).latent_dist.mode())
                th.cuda.empty_cache()
            z = (th.cat(zl) * 0.18215).cpu().numpy()
            del vae
            th.cuda.empty_cache()
            outd = (f"data/top10_style23/predskel_fmdit_seen20" if name == "seen20"
                    else f"data/top10_style23/predskel_fmdit_strict84")
            if a.dump_tag:
                outd += "_" + a.dump_tag
            os.makedirs(outd, exist_ok=True)
            np.savez_compressed(os.path.join(outd, "shard_00000.npz"),
                                latents=z.astype(np.float16),
                                img_ids=np.array(ids, dtype=np.int64))
            log(f"[dump] {name} -> {outd} ({len(ids)} 条) 预测墨比 {p64.mean():.4f}")
        return

    # ── 训练 ─────────────────────────────────────────────────────────────
    # ★ compile 必须放在 poster/dump 分支**之后**: 否则评测路径也会走编译图
    #   (load_state_dict -> 重新 trace, 又慢又吃显存)
    if a.compile:
        try:
            model = th.compile(model)
            log("[1b] torch.compile 已启用 (照搬主干 config: compile=true)")
        except Exception as _e:                    # noqa: BLE001
            log(f"[1b] torch.compile 失败, 回退 eager: {_e}")

    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    lf = open(a.log, "a", encoding="utf-8")

    def wlog(m):
        log(m)
        lf.write(m + "\n")
        lf.flush()

    best, bad, t0 = -1.0, 0, time.time()
    wlog(f"[2] 训练 {a.epochs} epochs = {total_steps} 步 batch={a.batch} "
         f"lr={a.lr} wd={a.wd} | struct w={a.w_struct} t>={a.struct_min_t} | "
         f"inkr w={a.w_ink_ratio} tol={a.ink_ratio_tol}")
    for step in range(1, total_steps + 1):
        idx = np.random.randint(0, len(tr_rows), a.batch)
        x0 = to_x(TR_T[idx]).unsqueeze(1).to(dev)
        g = to_x(TR_G[idx]).unsqueeze(1).to(dev)
        y = th.from_numpy(TR_Y[idx]).to(dev)
        t = sample_t(a.batch)
        eps = th.randn_like(x0)
        tc = t[:, None, None, None]
        xt = (1 - tc) * x0 + tc * eps
        v_tgt = eps - x0
        # ★ AMP(bf16) 前向: 实测纯 fp32 只有 0.86 TFLOPS (2.5s/step @ batch768),
        #   GPU 100%/395W 满载 -> 是算力瓶颈, 不是 overhead。loss 侧回 fp32 保精度。
        with th.autocast("cuda", dtype=th.bfloat16):
            v = vel(model, xt, t, g, y)
        v = v.float()
        l_mse = F.mse_loss(v, v_tgt)
        l_struct = th.zeros((), device=dev)
        l_inkr = th.zeros((), device=dev)
        if a.w_struct > 0 or a.w_ink_ratio > 0:
            sel = (t >= a.struct_min_t) & (t <= a.struct_max_t)
            if bool(sel.any()):
                tc_ = t[sel][:, None, None, None]
                x0p = xt[sel] - tc_ * v[sel]
                pin = ((1.0 - x0p) * 0.5).clamp(1e-4, 1 - 1e-4)
                tin = (1.0 - x0[sel]) * 0.5
                l_struct = soft_dice_prob(pin, tin) + 0.5 * F.binary_cross_entropy(
                    pin, tin)
                ps = pin.sum(dim=(1, 2, 3))
                ts = tin.sum(dim=(1, 2, 3)).clamp_min(1.0)
                l_inkr = (th.relu(ps - a.ink_ratio_tol * ts) / ts).mean()
        loss = l_mse + a.w_struct * l_struct + a.w_ink_ratio * l_inkr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with th.no_grad():
            for k, v_ in raw.state_dict().items():
                if v_.dtype.is_floating_point:
                    ema[k].mul_(a.ema_decay).add_(v_.detach(), alpha=1 - a.ema_decay)
                else:
                    ema[k].copy_(v_)
        if step % max(1, steps_per_epoch // 5) == 0:
            wlog(f"    epoch {step/steps_per_epoch:6.2f}  step {step:6d}  "
                 f"mse {float(l_mse):.5f}  struct {float(l_struct):.4f}  "
                 f"inkr {float(l_inkr):.4f}  {int(time.time()-t0)}s")
        if step % eval_every == 0:
            m = eval_val()
            wlog(f"    [eval] epoch {step//steps_per_epoch} (step {step}) | " +
                 "  ".join(f"{k} {v_:.4f}" for k, v_ in m.items()))
            if m["dice64"] > best:
                best, bad = m["dice64"], 0
                th.save({"ema": {k: v_.cpu() for k, v_ in ema.items()},
                         "step": step, "epoch": step // steps_per_epoch,
                         "args": vars(a), "metrics": m}, a.out)
                wlog(f"    [es] ★ 新最佳 {best:.4f} @ {step}")
            else:
                bad += 1
                wlog(f"    [es] 未改善 {bad}/{a.es_patience}")
                if bad >= a.es_patience:
                    wlog(f"    [es] ★★ 早停 @ {step} (best {best:.4f})")
                    break
        if a.save_every and step % a.save_every == 0:
            th.save({"ema": {k: v_.cpu() for k, v_ in ema.items()},
                     "step": step, "args": vars(a)}, a.out + f".step{step:06d}")
    wlog(f"[3] DONE best={best:.4f}")


if __name__ == "__main__":
    main()
