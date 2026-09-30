#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_skelnet_dit.py — 用一个小 DiT 生成「该书家写的那个字的骨架」latent。

任务: (标准骨架 g_std, 风格 e) -> 该书家真迹的骨架 latent (7px 载体)。
      与 DeformSkel(warp 头) 的关系: **并行方案**, 不是替换它的算子 ——
      这里不重采样任何东西, 直接从噪声生成目标, 因此不存在"细脊被摊断"的问题。

为什么是生成不是形变 (实测动机):
  · std 与 gt 可能只共享粗细、结构差很远 (简繁/异体/个人变体), 不是小形变;
  · 有界 warp (max_off=6 格) 走不完这段距离, 且散度大的地方必然断笔
    (32x32 上 3px 骨架只有 0.375 格: 位移 1 格 -> 墨量 0.58x / 连通 18x);
  · 纯 MSE 下「输出空白(0.379)」比「原样输出 std(0.416)」更接近 gt -> 回归目标病态。
  7px 载体把目标抬到 0.88 格, 监督信号才可分辨。

判据 (不用 ssim):
  ★ clDice: 生成骨架与 **gt** 骨架的中心线重合度 (结构性, 幅度无关)
  ★ idIoU: 生成骨架与 **std** 骨架的 IoU —— 防"幻觉成别的字"的一个粗查
  · 按字符留出 (assets/val_skelnet.csv) 才能测泛化

用法:
  python tools/train_skelnet_dit.py --steps 200 --smoke
  python tools/train_skelnet_dit.py --steps 40000
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

TIME_SCALE = 1000.0


# ── clDice / IoU (骨架域, 幅度无关) ────────────────────────────────────────
def _skel(b):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(b)
    except Exception:                                          # noqa: BLE001
        from scipy.ndimage import binary_erosion, generate_binary_structure
        st = generate_binary_structure(2, 2)
        sk = np.zeros_like(b)
        cur = b.copy()
        while cur.any():
            er = binary_erosion(cur, structure=st)
            sk |= cur & ~er
            cur = er
        return sk


def cldice_np(pred_b, gt_b, tol=3):
    """容差版 clDice: 中心线落在对方 **膨胀 tol 像素** 的范围内即算命中。

    ⚠ 必须带容差。精确像素重合 (`sp & gt_b`) 对 1px 中心线是灾难性的:
    同一個字、只是间架略不同, 交集也几乎为空 —— 实测 copy baseline
    (直接把输入标准骨架当预测) 的精确 clDice 只有 **0.045**, 完全不可用。
    这与仓库早已否决的 `skel_iou`(中位 0.017, "对 1px 偏移极敏感") 是同一个坑。

    tolerance 的尺度: 256px 图上 3px, 约等于骨架半个到一个笔宽。
    """
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    from scipy.ndimage import binary_dilation
    sp, sg = _skel(pred_b), _skel(gt_b)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    if tol > 0:
        st = np.ones((3, 3), bool)
        gt_t = binary_dilation(gt_b, structure=st, iterations=tol)
        pr_t = binary_dilation(pred_b, structure=st, iterations=tol)
    else:
        gt_t, pr_t = gt_b, pred_b
    tprec = float((sp & gt_t).sum()) / float(sp.sum())
    tsens = float((sg & pr_t).sum()) / float(sg.sum())
    if tprec + tsens <= 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="assets/train_top10_style23_minusval.csv")
    ap.add_argument("--val-csv", default="assets/val_skelnet.csv")
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std")
    ap.add_argument("--gt-png-dir", default="data/top10_style23/gt_skel_png")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--inject-layers", type=int, default=2,
                    help="标准骨架的逐层注入层数 (条件必须强, 它是唯一的字身份来源)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--style-drop", type=float, default=0.1, help="风格 dropout (CFG + 正则)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--sample-steps", type=int, default=20)
    ap.add_argument("--es-patience", type=int, default=8)
    ap.add_argument("--out", default="assets/skelnet_dit_v1.pt")
    ap.add_argument("--log", default="logs/skelnet_dit_v1.log")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pred", choices=["v", "x0"], default="v",
                    help="参数化: v=预测速度(flow, 主模型口径) | x0=直接预测干净样本。\n"
                         "本任务的 (字,书家) 配对近似确定性 -> x0 参数化可能收敛更快、"
                         "且单步即可出结果。")
    ap.add_argument("--bridge", action="store_true",
                    help="★ 以**标准骨架 g 为起点**做 rectified flow, 而不是从纯噪声起步。\n"
                         "  路径 xt=(1-t)*x0 + t*g, 速度目标 = x0-g (**就是形变残差**)。\n"
                         "  相对噪声起步的两大好处:\n"
                         "   (1) loss 直接监督残差, 不再被'重建输入'稀释 —— 噪声起步时\n"
                         "       cos(g,x0)=0.81, 模型只要'照抄'就能拿走大部分 loss, 是强吸引子;\n"
                         "       g 起步时'什么都不做'的 loss = ||x0-g||^2 很大, 不再是吸引子。\n"
                         "   (2) 字身份直接在输入流里, 不必靠弱的条件注入传递 -> 容量全花在形变,\n"
                         "       **不需要大模型**, 也降低过拟合风险。")
    ap.add_argument("--bridge-hide-g", action="store_true",
                    help="★ 与 --bridge 配合: **不给网络看 g** (喂零), 只把 g 当采样起点。\n"
                         "  ⚠ 不加这个会**泄漏**: xt=(1-t)*x0+t*g 且 g 又作为条件传入 ->\n"
                         "    模型可代数解出 x0=(xt-t*g)/(1-t), 不用学形变就能把 loss 压到 ~0。\n"
                         "    实测: 带 g 时 flow 0.003 但 resAlign 一路转负 (-0.60), 采样发散。\n"
                         "  隐藏 g 后该捷径不成立, 字身份只能从输入流 xt 里读, 才是真学形变。")
    ap.add_argument("--eval-only", action="store_true",
                    help="只载入 --resume 指定的 ckpt 跑一次验证, 打印全部守门指标后退出。")
    ap.add_argument("--resume", default="", help="配合 --eval-only 的 ckpt 路径")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    dev = a.device

    def log(msg):
        print(msg, flush=True)

    # ── 数据 ─────────────────────────────────────────────────────────────
    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = None
    if os.path.exists(a.callig_map):
        csmap = json.load(open(a.callig_map, encoding="utf-8"))
    # ⚠ 训练 csv 已经是「minusval」, 不能再按 val_ids 减一次 (那会把验证集减空)。
    #   验证集直接读单独的 val_skelnet.csv (与原训练 csv 同列, 只含留出行)。
    val_ids = set()
    if os.path.exists(a.val_csv):
        for r in csv.DictReader(open(a.val_csv, encoding="utf-8")):
            try:
                val_ids.add(int(os.path.basename(r.get("image_path", "")).split(".")[0]))
            except Exception:                                  # noqa: BLE001
                pass
    log(f"[1] 留出字符验证集 {len(val_ids)} 条 (来自 {a.val_csv})")

    def make_ds(csv_path, train):
        return MCCDLatentDataset(
            csv_file=csv_path, latent_shards_dir=a.tgt_shards, img_root="",
            image_size=256, is_train=train, preload=True, load_image=False,
            skel_latent_shards_dir=a.cond_shards,
            callig_id_map=None, callig_script_map=csmap)

    ds_tr = make_ds(a.csv, True)
    ds_va = make_ds(a.val_csv, False) if os.path.exists(a.val_csv) else None
    log(f"[1] 训练 {len(ds_tr)} / 验证 {0 if ds_va is None else len(ds_va)}")

    n_slots = int((csmap or {}).get("num_pairs", 0) or 0)
    if n_slots <= 0:
        n_slots = max(int(ds_tr[i]["y_callig"])
                      for i in range(min(256, len(ds_tr)))) + 1
    log(f"[1] 风格槽位 {n_slots}")

    # ── 模型 ─────────────────────────────────────────────────────────────
    from src.model.dit import DiT_2Cond
    model = DiT_2Cond(
        input_size=32, patch_size=2, in_channels=4, out_channels=4,
        depth=a.depth, hidden_size=a.hidden, num_heads=a.heads,
        num_calligraphers=n_slots, num_characters=1,
        use_char_cond=False, use_glyph_cond=True, glyph_in_channels=4,
        glyph_inject_layers=a.inject_layers, glyph_inject_mode="adaln",
        glyph_scale_init=0.6, glyph_drop_prob=0.0,
        glyph_embedder_depth=2,
        condition_fusion="factorized_cat", callig_embed_dim=128,
        glyph_vec_cond=True, glyph_vec_dim=128,
        cond_drop_all_prob=a.style_drop, cond_drop_one_prob=0.0,
        learn_sigma=False,
    ).to(dev)
    # 预训练风格表 (23,128)
    if os.path.exists(a.style_emb):
        d = th.load(a.style_emb, map_location="cpu", weights_only=False)
        emb = d["embedding"] if isinstance(d, dict) else d
        w = model.y_callig_embedder.embedding_table.weight
        if tuple(emb.shape) == (w.shape[0] - 1, w.shape[1]):
            with th.no_grad():
                w[:emb.shape[0]].copy_(emb.float())
            log(f"[2] 载入预训练风格表 {tuple(emb.shape)}")
        else:
            log(f"[2] ⚠ 风格表形状 {tuple(emb.shape)} != 模型 {tuple(w.shape)} 去 null 行, 跳过")
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[2] 模型 depth={a.depth} hidden={a.hidden} heads={a.heads} "
        f"inject={a.inject_layers} -> {n_par:,} 参数量")

    # step0 自检: 条件通路必须真的影响输出 (否则骨架通路是死的)。
    # ⚠ 必须先把 zero-init 的调制层激活, 否则整网输出结构性恒为 0 ——
    #   `adaLN_modulation[-1]` 与 `final_layer` 零初始化时, 任何"改条件应改输出"
    #   的测试都必然 Δ=0 (本仓库已踩过三次的同一个坑)。测完立刻恢复。
    _sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with th.no_grad():
        for blk in model.blocks:
            if hasattr(blk, "adaLN_modulation"):
                blk.adaLN_modulation[-1].weight.normal_(std=0.02)
                blk.adaLN_modulation[-1].bias.normal_(std=0.02)
        if hasattr(model.final_layer, "linear"):
            model.final_layer.linear.weight.normal_(std=0.02)
            model.final_layer.linear.bias.normal_(std=0.02)
        b0 = th.randn(2, 4, 32, 32, device=dev)
        t0 = th.full((2,), 500.0, device=dev)
        yc0 = th.zeros(2, dtype=th.long, device=dev)
        gA = th.randn(2, 4, 32, 32, device=dev)
        gB = th.randn(2, 4, 32, 32, device=dev)
        vA = model(b0, t0, y_callig=yc0, y_char=yc0, g=gA)
        vB = model(b0, t0, y_callig=yc0, y_char=yc0, g=gB)
        if isinstance(vA, tuple):
            vA, vB = vA[0], vB[0]
        d_cond = float((vA - vB).norm() / vA.norm().clamp_min(1e-8))
        vC = model(b0, t0, y_callig=th.ones(2, dtype=th.long, device=dev),
                   y_char=yc0, g=gA)[0]
        d_style = float((vA - vC).norm() / vA.norm().clamp_min(1e-8))
    model.load_state_dict(_sd0)
    log(f"[2a] 条件自检: 换 g 输出相对差 {d_cond:.4f} | 换风格 {d_style:.4f} "
        f"(>1e-3 说明通路接上了)")
    if a.smoke and (d_cond < 1e-3 or d_style < 1e-3):
        raise SystemExit("[FATAL] 骨架或风格条件没有影响输出 —— 通路没接上, 不要开训")

    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                         lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1), 1.0) *
        (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
         (1 + np.cos(np.pi * max(0.0, (s - a.warmup) /
                                 max(a.steps - a.warmup, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # VAE (只用于验证期的结构指标)
    vae = None
    if a.eval_every > 0:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)

    def load_batch(ds, idxs):
        xs, gs, ys, ids = [], [], [], []
        for i in idxs:
            b = ds[i]
            xs.append(b["latent"].float())
            gs.append(b["skel_latent"].float())
            ys.append(int(b["y_callig"]))
            ids.append(int(b["img_id"]))
        return (th.stack(xs).to(dev), th.stack(gs).to(dev),
                th.tensor(ys, device=dev), ids)

    def _cos(a_, b_):
        x, y = a_.ravel().float(), b_.ravel().float()
        return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))

    @th.no_grad()
    def eval_val():
        """按字符留出验证: 生成骨架 vs gt 骨架的 clDice (结构性判据, 幅度无关)。

        另记三个**守门指标** (只观测, 不进损失):
          · inkRatio = ink(生成)/ink(GT)      —— <0.5 白化 / >1.5 过墨, 都判死
          · cosGen   = cos(生成, GT) latent   —— 与 cosStd(照抄输入的 baseline) 比,
                       低于它就是"连照抄都不如", 说明形变没学到
        """
        model.eval()
        ck = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        n = min(a.val_n, len(ds_va)) if ds_va is not None else 0
        if n == 0:
            model.load_state_dict(ck)
            model.train()
            return {}
        from PIL import Image
        cls, idg = [], []
        inkp, inkg, cosgen, cosstd, resal = [], [], [], [], []
        B = 32
        for s in range(0, n, B):
            end = min(s + B, n)
            x0, g, y, ids = load_batch(ds_va, list(range(s, end)))
            z = g.clone() if a.bridge else th.randn_like(x0)   # ★ 起点
            src = g if a.bridge else z.clone()                 # x0 参数化需要固定参考
            ts = th.linspace(1.0, 0.0, a.sample_steps + 1, device=dev)
            for k in range(a.sample_steps):
                _gc = th.zeros_like(g) if (a.bridge and a.bridge_hide_g) else g
                out = model(z, th.full((z.shape[0],), float(ts[k]) * TIME_SCALE,
                                       device=dev),
                            y_callig=y, y_char=th.zeros_like(y), g=_gc)
                if isinstance(out, tuple):
                    out = out[0]
                if a.pred == "x0":
                    z = (1 - ts[k + 1]) * out + ts[k + 1] * src
                else:
                    z = z + (ts[k + 1] - ts[k]) * out
            dec = vae.decode(z / 0.18215).sample.mean(1)          # (b,256,256)
            pr = (dec < 0).cpu().numpy()                          # 墨 = 暗 = <0
            gdec = vae.decode(g / 0.18215).sample.mean(1)
            st = (gdec < 0).cpu().numpy()                         # 输入标准骨架
            for i in range(pr.shape[0]):
                fp = os.path.join(a.gt_png_dir, f"{ids[i]:06d}.png")
                if not os.path.exists(fp):
                    continue
                gt = np.asarray(Image.open(fp).convert("L")) < 128
                cls.append(cldice_np(pr[i], gt))
                u = (pr[i] | st[i]).sum()
                idg.append(float((pr[i] & st[i]).sum()) / u if u else 0.0)
                inkp.append(float(pr[i].mean()))
                inkg.append(float(gt.mean()))
                cosgen.append(_cos(z[i], x0[i]))
                cosstd.append(_cos(g[i], x0[i]))
                # ★ 残差对齐: 直接测"要学的那个量" (形变残差), 而不是"把输入重建出来"
                #   基线(照抄/什么都不做) = 0  |  上限 = 残差的组内一致度 ≈ 0.51
                #   (实测: 同一 (书家,字) 多张样本的残差方向余弦 +0.5087,
                #    不同 (书家,字) 之间 +0.0030 —— 见 tools/diag_residual_consistency.py)
                #   动态范围 0~0.51, 判别力远高于 clDice(基线0.27~上限~0.5)
                resal.append(_cos(z[i] - g[i], x0[i] - g[i]))
        model.load_state_dict(ck)
        model.train()
        out = {"clDice": float(np.mean(cls)) if cls else float("nan"),
               "idIoU": float(np.mean(idg)) if idg else float("nan")}
        if inkp:
            out["inkRatio"] = float(np.mean(inkp) / max(float(np.mean(inkg)), 1e-9))
            out["cosGen"] = float(np.mean(cosgen))
            out["cosStd"] = float(np.mean(cosstd))          # 照抄输入的 baseline
            out["resAlign"] = float(np.mean(resal))         # ★ 主判据
        return out

    # ── 训练 ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    logf = open(a.log, "a", encoding="utf-8")

    def wlog(m):
        log(m)
        logf.write(m + "\n")
        logf.flush()

    # ── 只评测模式: 载入 ckpt 的 EMA, 跑一次验证, 打印全部守门指标 ──
    if a.eval_only:
        if not a.resume:
            raise SystemExit("[FATAL] --eval-only 需要 --resume <ckpt>")
        _rf = th.load(a.resume, map_location="cpu", weights_only=False)
        ema.update({k: v for k, v in _rf["ema"].items()})
        wlog(f"[eval-only] 载入 {a.resume} (step={_rf.get('step')}, "
             f"tgt={a.tgt_shards})")
        m = eval_val()
        wlog(f"[eval-only] 指标: {m}")
        return

    best, bad = -1.0, 0
    t0 = time.time()
    wlog(f"[3] 训练 {a.steps} 步 batch={a.batch} lr={a.lr} wd={a.wd} "
         f"style_drop={a.style_drop}")
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(ds_tr), a.batch)
        x0, g, y, _ids = load_batch(ds_tr, idx)
        # ★ 起点: bridge 模式用**标准骨架 g**, 否则纯噪声。
        #   速度目标统一是 d(xt)/dt = src - x0 (t: 1->0 采样方向):
        #     噪声起步 -> eps - x0   (大部分在监督"重建输入")
        #     g  起步 -> g   - x0 = -(形变残差)  **直接监督要学的量**
        src = g if a.bridge else th.randn_like(x0)
        # bridge: 目标与 t 无关, 均匀覆盖整条直线即可;
        # 噪声:  用 logit-normal(0,1) (与主模型 train.py 的 t_sampler 一致) ——
        #   均匀采样把容量浪费在两端 (t→1 几乎是纯噪声不可预测, t→0 几乎就是数据)。
        if a.bridge:
            t = th.rand(x0.shape[0], device=dev).clamp(1e-3, 1 - 1e-3)
        else:
            t = th.sigmoid(th.randn(x0.shape[0], device=dev)).clamp(1e-3, 1 - 1e-3)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * src
        # ★ bridge 且 hide-g 时**不把 g 喂给网络** (喂零), 堵掉代数泄漏;
        #   字身份由输入流 xt 承载 (t=1 时 xt 就是 g)。
        g_cond = th.zeros_like(g) if (a.bridge and a.bridge_hide_g) else g
        v_pred = model(xt, t * TIME_SCALE, y_callig=y, y_char=th.zeros_like(y), g=g_cond)
        if isinstance(v_pred, tuple):
            v_pred = v_pred[0]
        if a.pred == "x0":
            loss = (v_pred - x0).pow(2).mean()
        else:
            loss = (v_pred - (src - x0)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 4 == 0:
            with th.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[k].copy_(v)
        if step % 50 == 0:
            wlog(f"    step {step:6d}  flow {float(loss):.4f}  "
                 f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s")
        if a.save_every > 0 and step % a.save_every == 0:
            sd = dict(deform=model.state_dict(), ema=ema, step=step,
                      args=vars(a), n_slots=n_slots)
            th.save(sd, a.out)
            th.save(sd, f"{a.out}.step{step:06d}")
            wlog(f"    [ckpt] {a.out}.step{step:06d}")
        if a.eval_every > 0 and step % a.eval_every == 0:
            m = eval_val()
            # ★ 早停/最佳 用 **残差对齐** (基线 0, 上限 ~0.51), 不用 clDice:
            #   clDice 的"什么都不做"基线就有 0.27, 动态范围被压死, 用它做早停等于盲选。
            sc = m.get("resAlign", m.get("clDice", float("nan")))
            _extra = ""
            if "inkRatio" in m:
                _flag = ""
                ir = m["inkRatio"]
                if ir < 0.5:
                    _flag = " ⚠白化"
                elif ir > 1.5:
                    _flag = " ⚠过墨"
                _extra = (f"  | resAlign {m['resAlign']:.4f} (基线0/上限~0.51)  "
                          f"clDice {m['clDice']:.4f}  ink比 {ir:.3f}{_flag}  "
                          f"cosGen {m['cosGen']:.4f} / cosStd {m['cosStd']:.4f}"
                          f" ({'优于' if m['cosGen'] > m['cosStd'] else '劣于'}照抄 "
                          f"{m['cosGen']-m['cosStd']:+.4f})")
            wlog(f"    [eval] step {step}{_extra}")
            if not np.isnan(sc):
                if sc > best + 2e-3:
                    best, bad = sc, 0
                    th.save(dict(deform=model.state_dict(), ema=ema, step=step,
                                 args=vars(a), n_slots=n_slots,
                                 best_cldice=sc), a.out + ".best")
                    wlog(f"    [es] ★ 新最佳 clDice {sc:.4f}")
                else:
                    bad += 1
                    wlog(f"    [es] 未改善 {bad}/{a.es_patience}")
                    if a.es_patience > 0 and bad >= a.es_patience:
                        wlog(f"    [es] ★★ 早停 @ {step} (best {best:.4f})")
                        break
    wlog(f"[4] 完成, best clDice = {best:.4f}")


if __name__ == "__main__":
    main()
