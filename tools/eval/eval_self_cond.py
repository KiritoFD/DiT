#!/usr/bin/env python3
"""A/B 测试: 普通采样 vs 自条件采样 (Self-Conditioning)。

在已有 12ch ckpt 上直接测试，无需重新训练。
比较同一组评测样本上的 SSIM / MSE / Skel Follow IoU。

用法 (远程 4090):
    python tools/eval/eval_self_cond.py \
        --ckpt assets/results/v11_pretrain_M432_adaln4_sym/checkpoints/<step>.pt \
        --config src/train/configs/v11_pretrain_M432_adaln4_sym.json \
        --eval-csv assets/eval_seen_v10.csv \
        --skel-dir data/skel/std_skel3_latents_fame_sym \
        --n 50 --device cuda \
        --first-pass-steps 20 \
        --blend-alphas 0.0,0.3,0.5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# 确保项目根目录在 path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.inference import (
    sample_latents, sample_latents_self_cond,
    load_eval_vae, build_diffusion, image_latent,
)


def load_model_from_ckpt(ckpt_path, config_path, device):
    """加载模型 + ckpt + 配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 构建模型
    from src.model import DiT_2Cond_models
    model_name = cfg.get('model', 'DiT-2Cond-S/2')
    _aux_dirs = [s for s in str(cfg.get('aux_latent_shards_dirs', '') or '').split(',') if s]
    in_channels = cfg.get('latent_channels', 4) + 4 * len(_aux_dirs)

    model = DiT_2Cond_models[model_name](
        num_calligraphers=cfg.get('num_calligraphers', 41),
        condition_fusion=cfg.get('condition_fusion', 'factorized_add'),
        callig_embed_dim=cfg.get('callig_embed_dim', 128),
        cond_drop_all_prob=0.0,
        cond_drop_one_prob=0.0,
        use_glyph_cond=True,
        use_char_cond=not cfg.get('no_char_cond', False),
        glyph_scale_init=cfg.get('glyph_scale_init', 0.4),
        glyph_drop_prob=0.0,
        glyph_inject_layers=cfg.get('glyph_inject_layers', 0),
        glyph_inject_mode=cfg.get('glyph_inject_mode', 'adaln'),
        glyph_embedder_depth=cfg.get('glyph_embedder_depth', 0),
        glyph_in_channels=4,
        in_channels=in_channels,
        norm_type=cfg.get('norm_type', 'rms'),
        mlp_type=cfg.get('mlp_type', 'swiglu'),
        qk_norm=bool(cfg.get('qk_norm', 1)),
        rope=bool(cfg.get('rope', 1)),
        rope_theta=cfg.get('rope_theta', 100.0),
        callig_proj_mode=cfg.get('callig_proj_mode', 'linear'),
        style_token_n=cfg.get('style_token_n', 0),
    )

    # 加载权重
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('ema', ckpt.get('model', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[ok] Model loaded: {model_name}, in_channels={in_channels}, "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, cfg


def load_eval_data(eval_csv, skel_dir, n, callig_id_map_path=None):
    """加载评测 CSV + 标准骨架 latent。"""
    import csv as csv_mod

    callig_map = {}
    if callig_id_map_path and os.path.exists(callig_id_map_path):
        with open(callig_id_map_path, 'r') as f:
            callig_map = json.load(f)

    rows = []
    with open(eval_csv, 'r', encoding='utf-8') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break

    conds, skels, gt_paths = [], [], []
    n_missing = 0
    for row in rows:
        callig_raw = int(row.get('calligrapher_id', row.get('callig_id', 0)))
        callig_id = int(callig_map.get(str(callig_raw), callig_raw))
        char_id = int(row.get('character_id', row.get('char_id', 0)))
        conds.append((callig_id, char_id))
        gt_paths.append(row.get('image_path', row.get('path', '')))

        img_id = row.get('img_id', row.get('image_id', ''))
        skel_path = os.path.join(skel_dir, f"{img_id}.npy")
        if os.path.exists(skel_path):
            skels.append(torch.from_numpy(np.load(skel_path)).float())
        else:
            skels.append(torch.zeros(4, 32, 32))
            n_missing += 1

    skels = torch.stack(skels)
    coverage = 1.0 - n_missing / len(conds) if conds else 0
    print(f"[ok] Loaded {len(conds)} eval samples, skel coverage: {coverage:.1%}")
    if coverage < 0.5:
        print(f"[WARN] Low skel coverage ({n_missing}/{len(conds)} missing). "
              f"Check skel_dir path.")
    return conds, skels, gt_paths


def compute_metrics(latents, gt_paths, vae, scaling_factor, device):
    """计算 SSIM 和 MSE (在像素空间)。"""
    from skimage.metrics import structural_similarity as ssim

    lat = image_latent(latents)
    results = []
    vae_dev = next(vae.parameters()).device

    for i in range(lat.shape[0]):
        with torch.no_grad():
            decoded = vae.decode(lat[i:i+1].to(vae_dev) / scaling_factor).sample
        pred = ((decoded[0].float().cpu().clamp(-1, 1) + 1) / 2).clamp(0, 1)
        pred_np = pred.permute(1, 2, 0).numpy()

        gt_path = gt_paths[i]
        if os.path.exists(gt_path):
            gt_img = np.array(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0
        else:
            results.append({'ssim': 0.0, 'mse': 1.0})
            continue

        _ssim = ssim(gt_img, pred_np, data_range=1.0, channel_axis=2)
        _mse = float(np.mean((gt_img - pred_np) ** 2))
        results.append({'ssim': _ssim, 'mse': _mse})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="A/B test: normal vs self-conditioning inference")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--skel-dir", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-scale", type=float, default=0.7)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--first-pass-steps", type=int, default=None,
                        help="ODE steps for pass 1 (None=same as pass 2)")
    parser.add_argument("--blend-alphas", type=str, default="0.0,0.3,0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--callig-id-map", type=str, default="assets/callig_id_map.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    blend_alphas = [float(x) for x in args.blend_alphas.split(',')]

    model, cfg = load_model_from_ckpt(args.ckpt, args.config, device)

    diffusion = build_diffusion(
        args.eval_steps, diffusion_type='flow',
        flow_kwargs={
            'sampler': cfg.get('flow_sampler', 'heun'),
            'heun_batch': True,
            'shift': cfg.get('shift', 1.0),
        })

    vae = load_eval_vae(device)
    scaling_factor = 0.18215

    conds, skels, gt_paths = load_eval_data(
        args.eval_csv, args.skel_dir, args.n, args.callig_id_map)

    torch.manual_seed(args.seed)
    in_ch = int(getattr(model, 'in_channels', 4))
    noise = torch.randn(len(conds), in_ch, 32, 32)

    # ── Baseline ──
    print(f"\n{'='*60}")
    print(f"  Baseline: Normal sampling ({args.eval_steps} Heun steps)")
    print(f"{'='*60}")
    t0 = time.time()
    lat_base = sample_latents(model, diffusion, noise, conds, args.cfg_scale,
                              args.batch, device, skel=skels, seed=args.seed)
    t_base = time.time() - t0
    m_base = compute_metrics(lat_base, gt_paths, vae, scaling_factor, device)
    ssim_base = np.mean([m['ssim'] for m in m_base])
    mse_base = np.mean([m['mse'] for m in m_base])
    print(f"  SSIM: {ssim_base:.4f}  MSE: {mse_base:.4f}  Time: {t_base:.1f}s")

    # ── Self-Conditioning ──
    results_summary = [{'alpha': 'baseline', 'ssim': ssim_base, 'mse': mse_base}]

    for alpha in blend_alphas:
        fps = args.first_pass_steps or args.eval_steps
        print(f"\n{'='*60}")
        print(f"  Self-Cond: alpha={alpha:.1f}, pass1={fps} steps, pass2={args.eval_steps} steps")
        print(f"{'='*60}")
        t0 = time.time()
        lat_sc = sample_latents_self_cond(
            model, diffusion, noise, conds, args.cfg_scale,
            args.batch, device, skel=skels, seed=args.seed,
            first_pass_steps=args.first_pass_steps,
            blend_alpha=alpha,
        )
        t_sc = time.time() - t0
        m_sc = compute_metrics(lat_sc, gt_paths, vae, scaling_factor, device)
        ssim_sc = np.mean([m['ssim'] for m in m_sc])
        mse_sc = np.mean([m['mse'] for m in m_sc])
        delta = ssim_sc - ssim_base

        improved = sum(1 for b, s in zip(m_base, m_sc) if s['ssim'] > b['ssim'])
        degraded = sum(1 for b, s in zip(m_base, m_sc) if s['ssim'] < b['ssim'])

        print(f"  SSIM: {ssim_sc:.4f} (delta {delta:+.4f})  MSE: {mse_sc:.4f}  Time: {t_sc:.1f}s")
        print(f"  Per-sample: {improved} improved, {degraded} degraded, "
              f"{len(m_base) - improved - degraded} tied")
        results_summary.append({'alpha': alpha, 'ssim': ssim_sc, 'mse': mse_sc,
                                'delta_ssim': delta, 'improved': improved, 'degraded': degraded})

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for r in results_summary:
        if r['alpha'] == 'baseline':
            print(f"  Baseline          SSIM={r['ssim']:.4f}  MSE={r['mse']:.4f}")
        else:
            print(f"  alpha={r['alpha']:<4}        SSIM={r['ssim']:.4f} ({r['delta_ssim']:+.4f})  "
                  f"MSE={r['mse']:.4f}  [{r['improved']}↑ {r['degraded']}↓]")


if __name__ == "__main__":
    main()
