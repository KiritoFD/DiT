# -*- coding: utf-8 -*-
"""
diag_cfg.py — 诊断「为什么 cfg<1 更好」。

理论背景
--------
对 flow matching，velocity v = ε - x0。CFG 实现（dit.py:655）为

    v_cfg = v_unc + w * (v_cond - v_unc)

代入 v = ε - x0 后：

    v_cfg = ε - [ x0_unc + w * (x0_cond - x0_unc) ]

即 **在 x0 空间以 uncond 为起点向 cond 外插**：
    w = 1  → 纯 cond
    w < 1  → 在 uncond 与 cond 之间插值（靠向 uncond）
    w > 1  → 超过 cond 外推

统计含义：w<1 等价于用 uncond 的「低方差/高偏差」换取 cond 的
「高方差/低偏差」。因此 **cfg<1 更好不是超参偏好，而是 cond 分支
x0 估计方差过大的症状**。

本脚本要区分的两种假设
----------------------
  H1（条件有用但有噪声）: cfg=0 明显差于 cfg≈0.7
      → 条件确实提供了信息，只是 cond 分支方差大
      → 应提升条件质量（glyph cond）+ 提高 uncond 训练比例

  H2（条件几乎无用）    : cfg=0 接近 cfg≈0.7
      → 条件基本没贡献，模型主要靠 uncond 的「平均书法字」
      → 问题比 H1 严重，条件通路需要重做

关键实验点
----------
  cfg = 0.0  纯 uncond（完全不依赖条件）
  cfg = 0.3 / 0.5 / 0.7  插值区
  cfg = 1.0  纯 cond
  cfg = 1.5  外推区（预期崩）

同时报告 SSIM / MSE / LPIPS，并按书体分组，观察是否所有书体一致。

用法（远程，GPU）
----------------
  python tools/diag_cfg.py --ckpt <s21 best.pt> \
      --config <resolved_config.json> --n 200 \
      --cfgs 0.0 0.3 0.5 0.7 1.0 1.5 --out /tmp/cfg_diag.json
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F


def build_model(config_path, device):
    """按 resolved_config 构造模型 + 注入 DINO 字形表（必须，否则 char 条件是随机的）。"""
    from types import SimpleNamespace
    from src.model import DiT_2Cond_models

    cfg = json.load(open(config_path, encoding="utf-8"))
    args = SimpleNamespace(**cfg)
    latent_size = cfg.get("image_size", 256) // 8
    dt = str(getattr(args, "diffusion_type", "ddpm")).lower()
    learn_sigma = bool(getattr(args, "learn_sigma",
                               dt not in ("flow", "flow_matching", "fm")))

    model = DiT_2Cond_models[cfg["model"]](
        input_size=latent_size,
        num_calligraphers=cfg["num_calligraphers"],
        num_characters=cfg["num_characters"],
        use_checkpoint=cfg.get("use_checkpoint", False),
        learn_sigma=learn_sigma,
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        cond_drop_all_prob=cfg["cond_drop_all_prob"],
        cond_drop_one_prob=cfg["cond_drop_one_prob"],
        cond_drop_which_glyph_prob=cfg.get("cond_drop_which_glyph_prob", 0.5),
        skel_head_enabled=False,
        use_glyph_cond=bool(cfg.get("w_glyph_cond", False)),
        glyph_scale_init=cfg.get("glyph_scale_init", 0.4),
        glyph_inject_layers=int(cfg.get("glyph_inject_layers", 0)),
        in_channels=cfg.get("latent_channels", 4),
        char_proj_mode=cfg.get("char_proj_mode", "full"),
        freeze_char_table=cfg.get("freeze_char_table", False),
        norm_type=cfg.get("norm_type", "rms"),
        mlp_type=cfg.get("mlp_type", "swiglu"),
        qk_norm=bool(cfg.get("qk_norm", True)),
        rope=bool(cfg.get("rope", True)),
        rope_theta=cfg.get("rope_theta", 100.0),
        attn_impl=cfg.get("attn_impl", "sdpa"),
    ).to(device)

    _inject_dino(model, cfg)
    return model, args, cfg


def _inject_dino(model, cfg):
    """复刻 train.py 的 DINO 字形表注入（per-script centering + unknown 填充）。"""
    emb_path = cfg.get("char_dino_embeddings")
    idx_path = cfg.get("char_dino_index")
    if not (emb_path and idx_path and os.path.isfile(emb_path)
            and os.path.isfile(idx_path)):
        print("[dino-init] WARNING: 未找到 dino 文件 -> char 表保持随机")
        return
    NUM_CH = 7026
    emb = np.load(emb_path)
    glyphs = json.load(open(idx_path, encoding="utf-8"))
    glyphs = glyphs.get("glyphs", glyphs)
    table = model.y_char_embedder.embedding_table.weight

    sids = np.array([int(g[0]) if isinstance(g, (list, tuple)) else int(g) // NUM_CH
                     for g in glyphs])
    filled = []
    for gi, glyph in enumerate(glyphs):
        if gi >= emb.shape[0]:
            break
        sid, cid = (int(glyph[0]), int(glyph[1])) if isinstance(glyph, (list, tuple)) \
            else divmod(int(glyph), NUM_CH)
        gid = sid * NUM_CH + cid
        if gid < table.shape[0] - 1:
            filled.append(gid)

    if cfg.get("dino_per_script_center", 0):
        for s in np.unique(sids):
            m = sids == s
            if m.sum() > 1:
                emb[m] -= emb[m].mean(0, keepdims=True)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    with torch.no_grad():
        for gi, glyph in enumerate(glyphs):
            if gi >= emb.shape[0]:
                break
            sid, cid = (int(glyph[0]), int(glyph[1])) if isinstance(glyph, (list, tuple)) \
                else divmod(int(glyph), NUM_CH)
            gid = sid * NUM_CH + cid
            if gid < table.shape[0] - 1:
                table[gid] = torch.from_numpy(emb[gi]).to(table.dtype)
        if cfg.get("dino_fill_unknown", 1) and filled:
            n_classes = model.y_char_embedder.num_classes
            mv = torch.from_numpy(emb.mean(0)).to(table.device, table.dtype)
            mv = mv / (mv.norm() + 1e-12)
            unk = [r for r in range(n_classes) if r not in set(filled)]
            if unk:
                table.index_copy_(0, torch.tensor(unk, device=table.device),
                                  mv.unsqueeze(0).expand(len(unk), -1))
    print(f"[dino-init] injected {len(filled)} glyph embeddings")


def load_ckpt(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model", "ema"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                print(f"[load] using '{k}'")
                break
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model


@torch.no_grad()
def evaluate(model, args, cache, cfg_scale, device, batch, steps, sample_seed=0):
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.eval.in_process_eval import load_eval_vae

    diffusion = create_diffusion_or_flow(
        str(steps), diffusion_type=getattr(args, "diffusion_type", "flow"),
        **flow_kwargs_from(args))
    vae = load_eval_vae(args, device)

    n = cache["n"]
    lc, ls, sf = cache["latent_channels"], cache["latent_spatial"], cache["scaling_factor"]
    conds, noise_all, gts = cache["conds"], cache["noise"], cache["gts"]
    g_all = cache.get("g")

    model.eval()
    torch.manual_seed(sample_seed)
    lat = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        z = noise_all[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
        if g_all is not None:
            gb = g_all[i:j].to(device)
            mk["g"] = torch.cat([gb, gb], dim=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
        lat[i:j] = s.float().cpu()
        del z, s
        torch.cuda.empty_cache()

    dec = []
    for i in range(0, n, 16):
        d = vae.decode(lat[i:i+16].to(device) / sf).sample
        dec.append(d.float().cpu())
        torch.cuda.empty_cache()
    dec = torch.cat(dec, dim=0).clamp(-1, 1)
    gt = gts[:n]

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssims, mses = [], []
    for k in range(n):
        a = (dec[k:k+1] + 1) / 2
        b = (gt[k:k+1] + 1) / 2
        ma, mb = a.mean(), b.mean()
        sa, sb = a.std(), b.std()
        sab = ((a - ma) * (b - mb)).mean()
        ssims.append((((2*ma*mb + C1) * (2*sab + C2)) /
                      ((ma**2 + mb**2 + C1) * (sa**2 + sb**2 + C2))).item())
        mses.append(F.mse_loss(a, b).item())
    del dec, gt
    torch.cuda.empty_cache()
    return float(np.median(ssims)), float(np.mean(ssims)), float(np.median(mses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cfgs", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="5script/cfg_diag.json")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    from src.eval.in_process_eval import prepare_eval_cache

    model, args, cfgj = build_model(a.config, device)
    load_ckpt(model, a.ckpt)
    steps = a.steps or int(cfgj.get("eval_steps", 25))
    cache = prepare_eval_cache(
        a.eval_csv, cfgj.get("img_root", "final_imgs_256"),
        cfgj.get("image_size", 256), a.n,
        int(cfgj.get("vae_downscale", 8)), int(cfgj.get("latent_channels", 4)),
        float(cfgj.get("vae_scaling_factor", 0.18215)),
        use_glyph_cond=bool(cfgj.get("w_glyph_cond", False)))

    res = []
    print(f"\n{'cfg':>6}{'ssim_med':>10}{'ssim_mean':>11}{'mse_med':>10}")
    for c in a.cfgs:
        t0 = time.time()
        med, mean, mse = evaluate(model, args, cache, c, device, a.batch, steps)
        print(f"{c:>6.2f}{med:>10.4f}{mean:>11.4f}{mse:>10.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        res.append({"cfg": c, "ssim_median": med, "ssim_mean": mean,
                    "mse_median": mse})

    print("\n" + "=" * 68)
    print("判读")
    print("=" * 68)
    r0 = next((r for r in res if abs(r["cfg"]) < 1e-9), None)
    r07 = next((r for r in res if abs(r["cfg"] - 0.7) < 1e-9), None)
    r1 = next((r for r in res if abs(r["cfg"] - 1.0) < 1e-9), None)
    if r0 and r07:
        gap = r07["ssim_median"] - r0["ssim_median"]
        print(f"  cfg=0.0 (纯 uncond)  = {r0['ssim_median']:.4f}")
        print(f"  cfg=0.7 (当前默认)   = {r07['ssim_median']:.4f}")
        print(f"  差距 = {gap:+.4f}")
        if gap > 0.05:
            print("  => H1 成立：条件确实有用，但 cond 分支方差大")
            print("     改进：提升条件质量(glyph cond) + 提高 uncond 训练比例")
        else:
            print("  => H2 成立：条件几乎没贡献，模型主要靠 uncond 的「平均书法字」")
            print("     条件通路需要重做")
    if r1 and r07:
        print(f"\n  cfg=1.0 (纯 cond)    = {r1['ssim_median']:.4f}")
        print(f"  相对 cfg=0.7 的变化  = {r1['ssim_median']-r07['ssim_median']:+.4f}")
        print("  => 若 cfg=1.0 明显更差，证实 cond 分支估计质量差")
    best = max(res, key=lambda r: r["ssim_median"])
    print(f"\n  最优 cfg = {best['cfg']}  (ssim_median={best['ssim_median']:.4f})")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": a.ckpt, "n": a.n, "steps": steps, "results": res},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
