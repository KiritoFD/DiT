# -*- coding: utf-8 -*-
"""
sweep_cfg_sharpness.py — 用**可计算指标**判定「cfg<1 是真实提升，还是变模糊骗分」。

为什么需要这个
--------------
观察：推理时 cfg<1 结果显著好。但从 CFG 语义看（flow 下在 x0 空间插值），
w<1 是让输出从 x0_cond 向 x0_uncond（"平均字"）回拉。

书法场景有一个陷阱，SSIM / LPIPS **都无法区分**：

  - 清晰但**写错**的字  -> SSIM 很低（结构完全不同）
  - **模糊**但轮廓接近平均字的字 -> SSIM 中等（轮廓大致对）

即 cfg<1 有可能只是把字变模糊，从而骗到更高 SSIM，而非真实质量提升。
人眼能分辨，但需要可计算的替代指标以便批量、可复现地判断。

三个互补指标
------------
1. **清晰度 (sharpness)**：拉普拉斯算子的方差。
   模糊图的高频成分少 -> 方差小。这是最直接的"变模糊"检测器。
2. **墨色占比 (ink_ratio)**：暗像素比例。
   若 cfg<1 让字变淡/变糊，墨色占比会下降（灰度被抹平）。
3. **二值 IoU (stroke_iou)**：把生成图与 GT 二值化后算 IoU。
   这衡量**笔画位置是否对**，对模糊敏感（糊了会同时降低 precision 和 recall）。

判读规则
--------
- 若 cfg 减小 -> sharpness 显著下降 且 SSIM 上升   => **变模糊骗分**（指标虚高）
- 若 cfg 减小 -> sharpness 基本不变 且 SSIM 上升   => **真实质量提升**
- 若 cfg 减小 -> sharpness 下降 但 stroke_iou 上升 => 结构更准，只是糊了（部分真实提升）

用法（远程）
-----------
  python tools/sweep_cfg_sharpness.py --ckpt <best.pt> --config <resolved.json> \
      --cfgs 0.0 0.3 0.5 0.7 1.0 --n 100 --out 5script/cfg_sharpness.csv
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F


# ---------------- 与 tools/sweep_cfg_visual.py 相同的模型构建 ----------------
def build_model_and_args(config_path, device):
    from types import SimpleNamespace
    from src.model import DiT_2Cond_models
    cfg = json.load(open(config_path, encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    _dt = str(getattr(args, 'diffusion_type', 'ddpm')).lower()
    _ls = getattr(args, 'learn_sigma', None)
    _learn_sigma = bool(_ls) if _ls is not None else _dt not in ('flow', 'flow_matching', 'fm')

    model = DiT_2Cond_models[cfg["model"]](
        input_size=cfg.get("image_size", 256) // 8,
        num_calligraphers=cfg["num_calligraphers"],
        num_characters=cfg["num_characters"],
        use_checkpoint=False, learn_sigma=_learn_sigma,
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        cond_drop_all_prob=0.0, cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=cfg.get("cond_drop_which_glyph_prob", 0.5),
        skel_head_enabled=False, use_glyph_cond=False,
        glyph_scale_init=cfg.get("glyph_scale_init", 0.4),
        in_channels=cfg.get("latent_channels", 4),
        char_proj_mode=cfg.get("char_proj_mode", "mlp"),
        freeze_char_table=cfg.get("freeze_char_table", False),
        norm_type=cfg.get("norm_type", "rms"),
        mlp_type=cfg.get("mlp_type", "swiglu"),
        qk_norm=bool(cfg.get("qk_norm", True)),
        rope=bool(cfg.get("rope", True)),
        rope_theta=cfg.get("rope_theta", 100.0),
        attn_impl=cfg.get("attn_impl", "sdpa"),
    ).to(device)
    _inject_dino(model, cfg, device)
    return model, args, cfg


def _inject_dino(model, cfg, device):
    emb_path = cfg.get("char_dino_embeddings")
    idx_path = cfg.get("char_dino_index")
    if not (emb_path and idx_path and os.path.isfile(emb_path) and os.path.isfile(idx_path)):
        print("[dino-init] WARNING: dino files missing -> char table random")
        return
    _NUM_CH = 7026
    _emb = np.load(emb_path)
    _glyphs = json.load(open(idx_path, encoding="utf-8"))
    _glyphs = _glyphs.get("glyphs", _glyphs)
    tbl = model.y_char_embedder.embedding_table.weight
    rows = []
    for gi, g in enumerate(_glyphs):
        if gi >= _emb.shape[0]:
            break
        sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) else divmod(int(g), _NUM_CH)
        gid = sid * _NUM_CH + cid
        if gid < tbl.shape[0] - 1:
            rows.append(gid)
    if cfg.get("dino_per_script_center", 0):
        sids = np.array([int(g[0]) if isinstance(g, (list, tuple)) else int(g) // _NUM_CH
                         for g in _glyphs])
        for s in np.unique(sids):
            m = sids == s
            if m.sum() > 1:
                _emb[m] -= _emb[m].mean(0, keepdims=True)
        _emb = _emb / np.maximum(np.linalg.norm(_emb, axis=1, keepdims=True), 1e-12)
    with torch.no_grad():
        for gi, g in enumerate(_glyphs):
            if gi >= _emb.shape[0]:
                break
            sid, cid = (int(g[0]), int(g[1])) if isinstance(g, (list, tuple)) else divmod(int(g), _NUM_CH)
            gid = sid * _NUM_CH + cid
            if gid < tbl.shape[0] - 1:
                tbl[gid] = torch.from_numpy(_emb[gi]).to(tbl.dtype)
        if cfg.get("dino_fill_unknown", 1) and rows:
            n = model.y_char_embedder.num_classes
            mv = torch.from_numpy(_emb.mean(0)).to(tbl.device, tbl.dtype)
            mv = mv / (mv.norm() + 1e-12)
            known = set(rows)
            unk = [r for r in range(n) if r not in known]
            if unk:
                tbl.index_copy_(0, torch.tensor(unk, device=tbl.device),
                                mv.unsqueeze(0).expand(len(unk), -1))
    print(f"[dino-init] injected {len(rows)} glyph embeddings")


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model", "ema"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                print(f"[load] using '{k}'")
                break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}")
    if miss:
        print(f"  missing sample: {miss[:4]}")


# ---------------- 指标 ----------------
_LAP = None


def laplacian_var(x):
    """拉普拉斯方差（清晰度）。x: (N,1,H,W) 灰度，值域 [0,1]。"""
    global _LAP
    if _LAP is None or _LAP.device != x.device:
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=x.device).view(1, 1, 3, 3)
        _LAP = k
    r = F.conv2d(x, _LAP, padding=1)
    return r.var(dim=(1, 2, 3))      # (N,)


def to_gray(t):
    """RGB [-1,1] -> 灰度 [0,1]（笔画暗、纸白）。"""
    g = (t.clamp(-1, 1) + 1) / 2     # [0,1]
    g = g.mean(dim=1, keepdim=True)  # (N,1,H,W)
    return g


def metrics(dec, gt):
    """返回 (ssim, mse, sharpness, ink_ratio, stroke_iou)。"""
    N = dec.shape[0]
    a = (dec + 1) / 2
    b = (gt[:N] + 1) / 2
    ssims, mses = [], []
    for k in range(N):
        x, y = a[k:k+1], b[k:k+1]
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()
        sxy = ((x - mx) * (y - my)).mean()
        ssims.append((((2*mx*my + C1) * (2*sxy + C2)) /
                      ((mx**2 + my**2 + C1) * (sx**2 + sy**2 + C2))).item())
        mses.append(F.mse_loss(x, y).item())

    gd = to_gray(dec)
    sharp = laplacian_var(gd)
    ink = (gd < 0.5).float().mean(dim=(1, 2, 3))        # 暗像素比例

    # 二值 IoU（笔画位置）
    gb = to_gray(gt[:N])
    pm = (gd < 0.5).float()
    gm = (gb < 0.5).float()
    inter = (pm * gm).sum(dim=(1, 2, 3))
    union = ((pm + gm) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1e-6)
    iou = (inter / union)

    return (float(np.median(ssims)), float(np.median(mses)),
            float(sharp.median()), float(ink.median()), float(iou.median()))


@torch.no_grad()
def sample_and_score(model, args, cache, cfg_scale, device, steps, dit_batch):
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.eval.in_process_eval import load_eval_vae
    diff = create_diffusion_or_flow(str(steps),
                                    diffusion_type=getattr(args, 'diffusion_type', 'flow'),
                                    **flow_kwargs_from(args))
    vae = load_eval_vae(args, device)
    n, lc, ls, sf = cache["n"], cache["latent_channels"], cache["latent_spatial"], cache["scaling_factor"]
    conds, noise = cache["conds"], cache["noise"]
    model.eval()
    torch.manual_seed(0)
    lat = torch.zeros(n, lc, ls, ls)
    for i in range(0, n, dit_batch):
        j = min(i + dit_batch, n)
        z = noise[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = diff.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                      clip_denoised=False,
                                      model_kwargs=dict(y_callig=yc, y_char=yh,
                                                        cfg_scale=cfg_scale),
                                      device=device)
        lat[i:j] = s.float().cpu()
        del z, s
        torch.cuda.empty_cache()
    outs = []
    for i in range(0, n, 16):
        d = vae.decode(lat[i:i+16].to(device) / sf).sample
        outs.append(d.float().cpu())
        torch.cuda.empty_cache()
    dec = torch.cat(outs).clamp(-1, 1)
    return metrics(dec, cache["gts"]) + (dec,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfgs", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="5script/cfg_sharpness.csv")
    ap.add_argument("--save-grid", default="", help="可选：另存对比网格图")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    from src.eval.in_process_eval import prepare_eval_cache
    model, args, cfgj = build_model_and_args(a.config, dev)
    load_ckpt(model, a.ckpt)
    steps = a.steps or int(cfgj.get("eval_steps", 25))
    cache = prepare_eval_cache(a.eval_csv, cfgj.get("img_root", "final_imgs_256"),
                               cfgj.get("image_size", 256), a.n,
                               int(cfgj.get("vae_downscale", 8)),
                               int(cfgj.get("latent_channels", 4)),
                               float(cfgj.get("vae_scaling_factor", 0.18215)))

    print(f"\n{'cfg':>6}{'SSIM':>9}{'MSE':>9}{'sharp':>11}{'ink%':>8}{'IoU':>9}")
    print("-" * 54)
    rows = []
    grids = {}
    for c in a.cfgs:
        t0 = time.time()
        ssim, mse, sharp, ink, iou, dec = sample_and_score(
            model, args, cache, c, dev, steps, a.batch)
        print(f"{c:>6.2f}{ssim:>9.4f}{mse:>9.4f}{sharp:>11.5f}"
              f"{ink*100:>8.2f}{iou:>9.4f}   ({time.time()-t0:.0f}s)", flush=True)
        rows.append({"cfg": c, "ssim": ssim, "mse": mse, "sharpness": sharp,
                     "ink_ratio": ink, "stroke_iou": iou})
        if a.save_grid:
            grids[c] = dec
        del dec
        torch.cuda.empty_cache()

    # 判读
    print("\n" + "=" * 60)
    print("判读")
    print("=" * 60)
    base = next((r for r in rows if abs(r["cfg"] - 1.0) < 1e-6), None)
    best = max(rows, key=lambda r: r["ssim"])
    if base:
        ds = best["ssim"] - base["ssim"]
        dsh = (best["sharpness"] - base["sharpness"]) / max(base["sharpness"], 1e-9)
        print(f"  最优 cfg = {best['cfg']}  (SSIM {best['ssim']:.4f})")
        print(f"  相对 cfg=1.0:  SSIM {ds:+.4f}   清晰度 {dsh*100:+.1f}%")
        if ds > 0.005 and dsh < -0.15:
            print("  => 变模糊骗分：SSIM 升但清晰度明显下降")
        elif ds > 0.005 and dsh > -0.05:
            print("  => 真实提升：SSIM 升且清晰度未降")
        else:
            print("  => 混合/不明显，建议结合 stroke_iou 与目检判断")
    for r in rows:
        print(f"    cfg={r['cfg']:<5} sharp={r['sharpness']:.5f} "
              f"ink={r['ink_ratio']*100:.2f}% IoU={r['stroke_iou']:.4f}")

    import csv as _csv
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["cfg", "ssim", "mse", "sharpness",
                                           "ink_ratio", "stroke_iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
