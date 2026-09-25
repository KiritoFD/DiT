#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""signal_tracing.py — 风格信号追踪 (关卡 3 / 4), 不训练。

在真实 ckpt 上一次前向/采样定位"风格进了 adaLN 却落不到像素"死在哪:

  3.1 Drop-g 差异放大
      固定 (字, 书体, 噪声), 比书家 A/B 的差异, 在 {有 g} 与 {g=0} 下。
      null/with > 1.5  => g 熨平了风格出口。
  3.3 逐层残差对比
      hook 每个 block 输出, ‖hA−hB‖/‖h‖。后 4 层 ≪ 前 4 层 => 深层抹平。
  4.1 Latent→Pixel 增益比
      Δz=‖zA−zB‖, Δx=‖dec(zA)−dec(zB)‖, Gain=Δx/Δz。
      对照: 给 zA 加同等范数的高斯噪声看 Δx_noise。
      Δz 已极小 => 死在上游; Δz 大但 Gain ≪ Gain_noise => VAE 吃掉差异。
  3.2 Loss 域梯度分解
      一个 batch, flow loss 按 GT 墨迹(膨胀 5px) / 白底拆开,
      各自对书家 embedding 的梯度范数。
      白底占比 > 80% 且墨迹梯度 ≪ 白底 => 白底稀释。

分组纪律: 必须同 (字, 书体) 不同书家。跨字比较测的是内容不是风格。
条件 id 用 dataset 的 y_callig。train.py 在设了 callig_script_map 时会把
num_calligraphers 改写成 pair 数(87), y_callig 就是这张表的行号。
y_callig_raw 是 45 人的书家索引, 用它会查到错误的行。

纯条件 (无 CFG), 固定噪声。判据是相对量。

    python tools/signal_tracing.py --ckpt <pt> --n-groups 24 --steps 20 --device cuda
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def unwrap(m):
    m = getattr(m, "module", m)
    return getattr(m, "_orig_mod", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-groups", type=int, default=24)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-grad", type=int, default=16)
    ap.add_argument("--probe-t", default="",
                    help="逗号分隔的 flow t。给定后不做采样, 只在这些 t 上测换书家的速度场差")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = a.device
    th.manual_seed(a.seed)
    np.random.seed(a.seed)

    from src.eval.model_io import load_model_from_ckpt
    from src.eval.inference import load_eval_vae
    from src.loss.flow_matching import TIME_SCALE

    model, args_ns = load_model_from_ckpt(a.ckpt, device=dev, use_ema=True, verbose=False)
    model.eval()
    raw = unwrap(model)
    lc = int(getattr(args_ns, "latent_channels", 4))
    vae = load_eval_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema")
    sf = float(getattr(args_ns, "vae_scaling_factor", 0.18215))
    n_cls = int(raw.y_callig_embedder.embedding_table.num_embeddings)

    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = None
    csm_path = getattr(args_ns, "callig_script_map", "") or ""
    if csm_path and os.path.exists(csm_path):
        csmap = json.load(open(csm_path, encoding="utf-8"))
    ds = MCCDLatentDataset(
        csv_file=args_ns.data_csv, latent_shards_dir=args_ns.latent_shards_dir,
        img_root=getattr(args_ns, "img_root", "") or "", image_size=256,
        preload=False, load_image=True,
        skel_latent_shards_dir=getattr(args_ns, "skel_latent_shards_dir", "") or None,
        callig_id_map=None, callig_script_map=csmap)

    # (字, 书体) -> 书家连续索引 -> 样本下标。同字同书体才是风格对照。
    by = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(ds.samples):
        raw_cid = int(r["calligrapher_id"])
        cm = (csmap or {}).get("callig_map") or {}
        cidx = int(cm.get(str(raw_cid), cm.get(raw_cid, raw_cid))) if cm else raw_cid
        by[(r["character"], int(r["script_id"]))][cidx].append(i)
    groups = []
    for key, calmap in by.items():
        if len(calmap) >= 2:
            cals = list(calmap)[:2]
            groups.append((key, [calmap[c][0] for c in cals], cals))
        if len(groups) >= a.n_groups:
            break
    print(f"[data] 组数={len(groups)} (同字同书体、不同书家) 表行数={n_cls} dev={dev}",
          flush=True)
    if len(groups) < 4:
        raise SystemExit("可用对照不足 4 组, 结果没有意义")

    def load_item(i):
        b = ds[i]
        yc = int(b["y_callig"])
        if not (0 <= yc < n_cls - 1):
            raise RuntimeError(f"y_callig={yc} 越出条件表 [0,{n_cls - 1})")
        # v17 配置 skel_as_glyph_cond=True: 训练把 batch['skel_latent']
        # (shards_std) 当作 g 传进模型。batch['g'] 是另一条 glyph 查表通路,
        # 这条配置里是空的, 用它会让"有 g"和"g=0"变成同一次采样。
        g = b["skel_latent"].float()
        if g.numel() == 0:
            g = b["g"].float()
        if g.numel() == 0:
            raise RuntimeError(f"样本 {i} 没有骨架条件")
        return (th.tensor([yc], device=dev),
                th.tensor([int(b["y_char"])], device=dev),
                g.to(dev)[None], b["image"], b["latent"].float())

    if a.probe_t:
        ts = [float(v) for v in a.probe_t.split(",") if v.strip()]
        acc_t = {tv: [] for tv in ts}
        for key, idxs, cals in groups:
            ycA, yhA, gA, _, lat = load_item(idxs[0])
            ycB, _, _, _, _ = load_item(idxs[1])
            x0 = lat.to(dev)[None]
            eps = th.randn_like(x0)
            for tv in ts:
                xt = (1 - tv) * x0 + tv * eps
                tt = th.full((1,), tv, device=dev) * TIME_SCALE
                with th.no_grad():
                    vA = model(xt, tt, y_callig=ycA, y_char=yhA, g=gA)
                    vB = model(xt, tt, y_callig=ycB, y_char=yhA, g=gA)
                if isinstance(vA, tuple):
                    vA = vA[0]
                if isinstance(vB, tuple):
                    vB = vB[0]
                acc_t[tv].append(float((vA - vB).norm() / (vA.norm() + 1e-6)))
        print("\n================ 时间门: 换书家的速度场差 ================", flush=True)
        for tv in ts:
            print(f"  t={tv:.2f}  ‖vA-vB‖/‖vA‖ = {float(np.mean(acc_t[tv])):.4f}")
        if a.out:
            json.dump({"ckpt": a.ckpt, "tag": a.tag,
                       "vel_style": {str(k): float(np.mean(v)) for k, v in acc_t.items()}},
                      open(a.out, "w", encoding="utf-8"), indent=2)
        print(f"[done] {a.ckpt}", flush=True)
        return

    def sample(yc, yh, g, eps):
        x = eps.clone()
        ts = th.linspace(1.0, 0.0, a.steps + 1)
        hs = {}

        def mk(idx):
            def h(_m, _i, out):
                hs[idx] = out.detach()
            return h

        handles = [raw.blocks[k].register_forward_hook(mk(k)) for k in range(len(raw.blocks))]
        with th.no_grad():
            for k in range(a.steps):
                t = float(ts[k])
                dt = float(ts[k + 1] - ts[k])
                v = model(x, th.full((1,), t, device=dev) * TIME_SCALE,
                           y_callig=yc, y_char=yh, g=g)
                if isinstance(v, tuple):
                    v = v[0]
                x = x + dt * v
        for hnd in handles:
            hnd.remove()
        return x, hs

    def dec(z):
        with th.no_grad():
            return vae.decode(z[:, :lc] / sf).sample

    acc = {k: [] for k in ("pix_with", "pix_null", "z_with", "z_null", "gain", "gain_noise")}
    per_block = [[] for _ in range(len(raw.blocks))]
    for gi, (key, idxs, cals) in enumerate(groups):
        ycA, yhA, gA, _, _ = load_item(idxs[0])
        ycB, _, _, _, _ = load_item(idxs[1])
        eps = th.randn(1, lc, 32, 32, device=dev)
        zA, hA = sample(ycA, yhA, gA, eps)
        zB, hB = sample(ycB, yhA, gA, eps)
        gz = th.zeros_like(gA)
        zA0, _ = sample(ycA, yhA, gz, eps)
        zB0, _ = sample(ycB, yhA, gz, eps)
        xA, xB = dec(zA), dec(zB)
        xA0, xB0 = dec(zA0), dec(zB0)

        def rel(u, v):
            return float((u - v).norm() / (u.norm() + 1e-6))

        acc["pix_with"].append(rel(xA, xB))
        acc["pix_null"].append(rel(xA0, xB0))
        acc["z_with"].append(rel(zA, zB))
        acc["z_null"].append(rel(zA0, zB0))
        dz = float((zA - zB).norm())
        acc["gain"].append(float((xA - xB).norm()) / (dz + 1e-6))
        direction = th.randn_like(zA)
        direction = direction / (direction.norm() + 1e-6) * dz
        acc["gain_noise"].append(float((dec(zA + direction) - xA).norm()) / (dz + 1e-6))
        for bi in range(len(raw.blocks)):
            ha, hb = hA[bi], hB[bi]
            per_block[bi].append(float((ha - hb).norm()) / (float(ha.norm()) + 1e-6))
        if gi % 4 == 0:
            print(f"  group {gi}/{len(groups)} {key} calligs={cals}", flush=True)

    def mean(k):
        return float(np.mean(acc[k]))

    pb = [float(np.mean(v)) for v in per_block]
    out = {
        "ckpt": a.ckpt, "tag": a.tag, "n_groups": len(groups), "steps": a.steps,
        "pix_with": mean("pix_with"), "pix_null": mean("pix_null"),
        "null_over_with_pix": mean("pix_null") / max(mean("pix_with"), 1e-9),
        "z_with": mean("z_with"), "z_null": mean("z_null"),
        "null_over_with_z": mean("z_null") / max(mean("z_with"), 1e-9),
        "gain": mean("gain"), "gain_noise": mean("gain_noise"),
        "gain_over_noise": mean("gain") / max(mean("gain_noise"), 1e-9),
        "per_block": pb,
        "early4": float(np.mean(pb[:4])), "late4": float(np.mean(pb[-4:])),
    }

    print("\n================ 关卡3.1 Drop-g 差异放大 ================", flush=True)
    print(f"  风格像素差 (有 g) = {out['pix_with']:.4f}")
    print(f"  风格像素差 (g=0)  = {out['pix_null']:.4f}")
    print(f"  放大比 null/with  = {out['null_over_with_pix']:.2f}x"
          f"   (>1.5 => g 熨平风格出口)")
    print(f"  [latent] 有g={out['z_with']:.4f}  g=0={out['z_null']:.4f}"
          f"  比={out['null_over_with_z']:.2f}x")

    print("\n================ 关卡4.1 Latent→Pixel 增益比 ================", flush=True)
    print(f"  Gain(风格)   = {out['gain']:.3f}")
    print(f"  Gain(等范数噪声) = {out['gain_noise']:.3f}")
    print(f"  风格/噪声     = {out['gain_over_noise']:.2f}"
          f"   (≪1 => VAE 吃掉风格差异; Δz 本身小 => 死在上游)")

    print("\n================ 关卡3.3 逐层残差 ‖hA−hB‖/‖h‖ ================", flush=True)
    for bi, v in enumerate(pb):
        print(f"    block {bi:>2}: {v:.4f}  " + "#" * int(v * 200))
    print(f"  前4层={out['early4']:.4f}  后4层={out['late4']:.4f}"
          f"  后/前={out['late4'] / max(out['early4'], 1e-9):.2f}"
          f"   (后≪前 => 深层抹平)")

    # ── 3.2 一个 batch 的损失域梯度 ──
    try:
        from scipy.ndimage import binary_dilation
        pick = [groups[i][1][0] for i in range(min(a.n_grad, len(groups)))]
        items = [load_item(i) for i in pick]
        yc = th.cat([it[0] for it in items])
        yh = th.cat([it[1] for it in items])
        g = th.cat([it[2] for it in items])
        x0 = th.stack([it[4] for it in items]).to(dev)
        masks = []
        for it in items:
            im = it[3].float().mean(0).numpy()
            ink = binary_dilation(im < 128, iterations=5)
            mt = th.from_numpy(ink.astype("float32"))[None, None]
            masks.append(th.nn.functional.max_pool2d(mt, 8)[0, 0])
        mask = th.stack(masks).to(dev)
        # freeze_callig_table 会把整表 requires_grad 关掉、把 null 行拆成
        # 独立 Parameter。诊断要的是"损失对书家行的敏感度"，所以临时打开。
        emb = raw.y_callig_embedder.embedding_table
        emb.weight.requires_grad_(True)
        eps = th.randn_like(x0)
        t = th.full((x0.shape[0],), 0.5, device=dev)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
        v = model(xt, t * TIME_SCALE, y_callig=yc, y_char=yh, g=g)
        if isinstance(v, tuple):
            v = v[0]
        per = (v - (eps - x0)).pow(2).mean(1)
        ink_loss = (per * mask).sum() / mask.sum().clamp_min(1)
        white_loss = (per * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1)
        n_pix = float(per[0].numel())
        ink_share = float(mask.sum() / mask.numel())
        white_share = 1.0 - ink_share
        grad_ink = th.autograd.grad(ink_loss, emb.weight, retain_graph=True)[0]
        grad_white = th.autograd.grad(white_loss, emb.weight)[0]
        gi = float(grad_ink.norm())
        gw = float(grad_white.norm())
        out.update({
            "ink_area_share": ink_share, "white_area_share": white_share,
            "grad_ink": gi, "grad_white": gw,
            "grad_ink_over_white": gi / max(gw, 1e-12),
            "grad_ink_per_area": gi / max(ink_share, 1e-6),
            "grad_white_per_area": gw / max(white_share, 1e-6),
            "n_pix": n_pix,
        })
        print("\n================ 关卡3.2 Loss 域梯度分解 ================", flush=True)
        print(f"  墨迹面积占比          = {ink_share * 100:.1f}%")
        print(f"  白底面积占比          = {white_share * 100:.1f}%")
        print(f"  ‖dL_ink / d table‖   = {gi:.5f}")
        print(f"  ‖dL_white / d table‖ = {gw:.5f}")
        print(f"  墨迹/白底 梯度比       = {out['grad_ink_over_white']:.3f}")
        print(f"  单位面积梯度 墨/白     = "
              f"{out['grad_ink_per_area'] / max(out['grad_white_per_area'], 1e-12):.2f}x"
              f"   (≈1 => 没有被稀释, 只是面积; ≪1 => 墨迹梯度真被淹没)")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"\n[3.2 失败] {type(ex).__name__}: {ex}", flush=True)

    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n[saved] {a.out}", flush=True)
    print(f"[done] {a.ckpt}", flush=True)


if __name__ == "__main__":
    main()
