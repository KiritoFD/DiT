# -*- coding: utf-8 -*-
"""
sweep_eval_cfg.py — 扫描 eval_cfg 对 base（无骨架）生成质量的影响。

动机
----
当前所有 fame 评测统一用 cfg=0.7，理由见 docs/system/13_fame_zero_shot.md §1.3：
「骨架条件下 CFG>1 有害，单调劣化」。

但 cfg<1 在 CFG 语义上是**削弱条件、向无条件分布靠拢**。这个 0.7 是为
「有骨架」场景调出来的，而 base 是**无骨架**场景 —— 两者对 CFG 的最优值
未必相同。若 base 在 cfg=0.7 下被人为压低，那么「预训练差得出奇 (0.50)
vs 后训练不错 (0.80)」这个对比本身就是失真的，后续所有基于 0.50 的
归因（如"char 条件是瓶颈"）都要重新审视。

本脚本不训练，只加载已有 ckpt 扫一组 cfg，直接算 SSIM/MSE。

用法（远程）
-----------
  python tools/sweep_eval_cfg.py --ckpt <ckpt.pt> --config <resolved_config.json> \
      --n 200 --cfgs 0.7 1.0 1.5 2.0 3.0 --out /tmp/cfg_sweep.json
"""
import os, sys, json, argparse, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_model_and_args(config_path, device):
    """按 resolved_config 构造模型，并复用 train.py 的 DINO 注入逻辑。"""
    from types import SimpleNamespace
    from src.model import DiT_2Cond_models

    cfg = json.load(open(config_path, encoding="utf-8"))
    args = SimpleNamespace(**cfg)

    latent_size = cfg.get("image_size", 256) // 8
    _diffusion_type = str(getattr(args, 'diffusion_type', 'ddpm')).lower()
    _ls = getattr(args, 'learn_sigma', None)
    _learn_sigma = bool(_ls) if _ls is not None else \
        _diffusion_type not in ('flow', 'flow_matching', 'fm')

    model = DiT_2Cond_models[cfg["model"]](
        input_size=latent_size,
        num_calligraphers=cfg["num_calligraphers"],
        num_characters=cfg["num_characters"],
        use_checkpoint=cfg.get("use_checkpoint", False),
        learn_sigma=_learn_sigma,
        condition_fusion=cfg["condition_fusion"],
        callig_embed_dim=cfg["callig_embed_dim"],
        char_embed_dim=cfg["char_embed_dim"],
        cond_drop_all_prob=cfg["cond_drop_all_prob"],
        cond_drop_one_prob=cfg["cond_drop_one_prob"],
        cond_drop_which_glyph_prob=cfg.get("cond_drop_which_glyph_prob", 0.5),
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=cfg.get("glyph_scale_init", 0.4),
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

    # ── 复用 train.py 的 DINO glyph-embedding 注入（否则 char 表是随机初始化）──
    _inject_dino(model, cfg, device)

    return model, args, cfg


def _inject_dino(model, cfg, device):
    """复刻 train.py:270-361 的 DINO 注入 + per-script centering + unknown 填充。

    必须与训练时完全一致，否则 char 条件是随机噪声，扫出来的 cfg 结论无意义。
    """
    import re
    emb_path = cfg.get("char_dino_embeddings")
    idx_path = cfg.get("char_dino_index")
    if not (emb_path and idx_path and os.path.isfile(emb_path)
            and os.path.isfile(idx_path)):
        print("[dino-init] WARNING: dino files not found -> char table stays random")
        return
    _NUM_CH = 7026
    _emb = np.load(emb_path)
    _glyphs = json.load(open(idx_path, encoding="utf-8"))
    _glyphs = _glyphs.get("glyphs", _glyphs)
    _table = model.y_char_embedder.embedding_table.weight

    loaded = 0
    filled_rows = []
    for gi, glyph in enumerate(_glyphs):
        if isinstance(glyph, (list, tuple)):
            sid, cid = int(glyph[0]), int(glyph[1])
        else:
            sid, cid = divmod(int(glyph), _NUM_CH)
        gid = sid * _NUM_CH + cid
        if gid < _table.shape[0] - 1 and gi < _emb.shape[0]:
            filled_rows.append(gid)
            loaded += 1

    if cfg.get("dino_per_script_center", 0):
        sids = np.array([int(g[0]) if isinstance(g, (list, tuple))
                         else int(g) // _NUM_CH for g in _glyphs])
        for s in np.unique(sids):
            m = sids == s
            if m.sum() > 1:
                _emb[m] -= _emb[m].mean(0, keepdims=True)
        _n = np.linalg.norm(_emb, axis=1, keepdims=True)
        _emb = _emb / np.maximum(_n, 1e-12)

    with torch.no_grad():
        for gi, glyph in enumerate(_glyphs):
            if gi >= _emb.shape[0]:
                break
            if isinstance(glyph, (list, tuple)):
                sid, cid = int(glyph[0]), int(glyph[1])
            else:
                sid, cid = divmod(int(glyph), _NUM_CH)
            gid = sid * _NUM_CH + cid
            if gid < _table.shape[0] - 1:
                _table[gid] = torch.from_numpy(_emb[gi]).to(_table.dtype)

        if cfg.get("dino_fill_unknown", 1) and filled_rows:
            n_classes = model.y_char_embedder.num_classes
            mean_vec = torch.from_numpy(_emb.mean(0)).to(_table.device, _table.dtype)
            mean_vec = mean_vec / (mean_vec.norm() + 1e-12)
            known = set(filled_rows)
            unk = [r for r in range(n_classes) if r not in known]
            if unk:
                idx_t = torch.tensor(unk, device=_table.device)
                _table.index_copy_(0, idx_t, mean_vec.unsqueeze(0).expand(len(unk), -1))
    print(f"[dino-init] injected {loaded} glyph embeddings "
          f"(per_script_center={cfg.get('dino_per_script_center', 0)}, "
          f"fill_unknown={cfg.get('dino_fill_unknown', 1)})")


def load_ckpt(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        for key in ("model", "ema"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                print(f"[load] using '{key}' weights")
                break
    # 去掉 DDP 前缀
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing sample: {missing[:5]}")


@torch.no_grad()
def evaluate_cfg(model, args, cache, cfg_scale, device, dit_batch, steps):
    """严格对齐 run_gpu_eval 的调用方式：ddim_sample_loop + model_kwargs。"""
    from src.loss import create_diffusion_or_flow, flow_kwargs_from
    from src.eval.in_process_eval import load_eval_vae

    diffusion = create_diffusion_or_flow(
        str(steps),
        diffusion_type=getattr(args, 'diffusion_type', 'flow'),
        **flow_kwargs_from(args),
    )
    vae = load_eval_vae(args, device)

    n = cache["n"]
    lc = cache["latent_channels"]
    ls = cache["latent_spatial"]
    sf = cache["scaling_factor"]
    conds = cache["conds"]
    noise_all = cache["noise"]
    gts_all = cache["gts"]

    model.eval()
    torch.manual_seed(0)
    all_latents = torch.zeros(n, lc, ls, ls, dtype=torch.float32)
    for i in range(0, n, dit_batch):
        j = min(i + dit_batch, n)
        z = noise_all[i:j].to(device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[1] for c in conds[i:j]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg_scale)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device,
            )
        all_latents[i:j] = samples.float().cpu()
        del z, samples
        torch.cuda.empty_cache()

    dec_out = []
    for i in range(0, n, 16):
        x = all_latents[i:i+16].to(device)
        d = vae.decode(x / sf).sample
        dec_out.append(d.float().cpu())
        del x, d
        torch.cuda.empty_cache()
    dec = torch.cat(dec_out, dim=0).clamp(-1, 1)

    gt = gts_all[:n]
    ssims, mses = [], []
    for k in range(n):
        a = (dec[k:k+1] + 1) / 2
        b = (gt[k:k+1] + 1) / 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_a, mu_b = a.mean(), b.mean()
        sa, sb = a.std(), b.std()
        sab = ((a - mu_a) * (b - mu_b)).mean()
        ssim = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sa ** 2 + sb ** 2 + C2))
        ssims.append(ssim.item())
        mses.append(F.mse_loss(a, b).item())
    del dec, gt
    torch.cuda.empty_cache()
    return float(np.median(ssims)), float(np.mean(ssims)), float(np.median(mses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="resolved_config.json")
    ap.add_argument("--eval-csv", default="5script/eval_fame_strict.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cfgs", type=float, nargs="+",
                    default=[0.7, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--steps", type=int, default=0, help="0=用 config 的 eval_steps")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="/tmp/cfg_sweep.json")
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    from src.eval.in_process_eval import prepare_eval_cache

    model, targs, cfgj = build_model_and_args(args_cli.config, device)
    load_ckpt(model, args_cli.ckpt)

    steps = args_cli.steps or int(cfgj.get("eval_steps", 25))
    cache = prepare_eval_cache(
        args_cli.eval_csv, cfgj.get("img_root", "final_imgs_256"),
        cfgj.get("image_size", 256), args_cli.n,
        int(cfgj.get("vae_downscale", 8)), int(cfgj.get("latent_channels", 4)),
        float(cfgj.get("vae_scaling_factor", 0.18215)))

    results = []
    print(f"\n{'cfg':>6}{'ssim_med':>10}{'ssim_mean':>11}{'mse_med':>10}")
    for c in args_cli.cfgs:
        t0 = time.time()
        med, mean, mse = evaluate_cfg(model, targs, cache, c, device,
                                      args_cli.batch, steps)
        print(f"{c:>6.2f}{med:>10.4f}{mean:>11.4f}{mse:>10.4f}  ({time.time()-t0:.0f}s)",
              flush=True)
        results.append({"cfg": c, "ssim_median": med,
                        "ssim_mean": mean, "mse_median": mse})

    best = max(results, key=lambda r: r["ssim_median"])
    print(f"\nbest cfg = {best['cfg']} (ssim_median={best['ssim_median']:.4f})")
    base = next((r for r in results if abs(r["cfg"] - 0.7) < 1e-6), None)
    if base:
        print(f"cfg=0.7 基线 = {base['ssim_median']:.4f}  "
              f"→ 最优提升 {best['ssim_median'] - base['ssim_median']:+.4f}")

    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args_cli.ckpt, "n": args_cli.n, "steps": steps,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args_cli.out}")


if __name__ == "__main__":
    main()
