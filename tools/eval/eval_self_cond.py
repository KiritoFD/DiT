#!/usr/bin/env python3
"""A/B 测试: 普通采样 vs 自条件采样 (Self-Conditioning)。

在已有 12ch ckpt 上直接测试，无需重新训练。
统一使用 make_eval_cache 加载 shard 数据，保证与官方评测完全同口径。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# 确保项目根目录在 path 中
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.eval.inference import (
    make_eval_cache, sample_latents, sample_latents_self_cond,
    load_eval_vae, build_diffusion, image_latent,
)
from src.utils.callig_map import load_callig_id_map


def load_model_from_ckpt(ckpt_path, config_path, device):
    """加载模型 + ckpt + 配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

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

    if cfg.get("freeze_callig_table"):
        model.y_callig_embedder.freeze_table()

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


def decode_latents_to_images(latents, vae, scaling_factor):
    """把 latents 解码为 [0, 1] 范围的 numpy 图像 (N, H, W, 3)。"""
    lat = image_latent(latents)
    vae_dev = next(vae.parameters()).device
    images = []
    batch_size = 16
    for i in range(0, lat.shape[0], batch_size):
        j = min(i + batch_size, lat.shape[0])
        with torch.no_grad():
            dec = vae.decode(lat[i:j].to(vae_dev) / scaling_factor).sample
        preds = ((dec.float().cpu().clamp(-1, 1) + 1) / 2).clamp(0, 1)
        for p in preds:
            images.append(p.permute(1, 2, 0).numpy())
    return images


def compute_metrics(pred_imgs, gt_tensors):
    """计算 SSIM 和 MSE。gt_tensors 为 [-1, 1] 的 torch.Tensor (N, 3, H, W)。"""
    results = []
    for i in range(len(pred_imgs)):
        pred = pred_imgs[i]
        gt = ((gt_tensors[i].clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
        _s = ssim(gt, pred, data_range=1.0, channel_axis=2)
        _m = float(np.mean((gt - pred) ** 2))
        results.append({'ssim': _s, 'mse': _m})
    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test: normal vs self-conditioning inference")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--skel-dir", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-scale", type=float, default=0.7)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--first-pass-steps", type=int, default=None)
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
    scaling_factor = float(cfg.get('vae_scaling_factor', 0.18215))

    # 加载 callig map
    cmap = None
    if args.callig_id_map and os.path.exists(args.callig_id_map):
        cmap, _ = load_callig_id_map(args.callig_id_map)

    # 统一使用 make_eval_cache
    img_root = cfg.get("gpu_eval_img_root") or cfg.get("img_root")
    gts, conds, _, skels_latent, noise = make_eval_cache(
        args.eval_csv, img_root, None, 256, args.n, 8, 4, scaling_factor,
        skel_latent_shards_dir=args.skel_dir, callig_id_map=cmap,
    )

    coverage = (skels_latent.abs().sum(dim=(1, 2, 3)) > 0).float().mean() if skels_latent is not None else 0
    print(f"[ok] Loaded {len(conds)} samples via make_eval_cache, skel coverage: {coverage:.1%}")

    # 将 noise 扩展到模型通道数
    in_ch = int(getattr(model, 'in_channels', 4))
    if noise.shape[1] < in_ch:
        torch.manual_seed(args.seed)
        extra = torch.randn(noise.shape[0], in_ch - noise.shape[1], *noise.shape[2:])
        noise = torch.cat([noise, extra], dim=1)

    # ── Baseline ──
    print(f"\n{'='*60}")
    print(f"  Baseline: Normal sampling ({args.eval_steps} Heun steps)")
    print(f"{'='*60}")
    t0 = time.time()
    lat_base = sample_latents(model, diffusion, noise, conds, args.cfg_scale,
                              args.batch, device, skel=skels_latent, seed=args.seed)
    t_base = time.time() - t0
    imgs_base = decode_latents_to_images(lat_base, vae, scaling_factor)
    m_base = compute_metrics(imgs_base, gts)
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
            args.batch, device, skel=skels_latent, seed=args.seed,
            first_pass_steps=args.first_pass_steps,
            blend_alpha=alpha,
        )
        t_sc = time.time() - t0
        imgs_sc = decode_latents_to_images(lat_sc, vae, scaling_factor)
        m_sc = compute_metrics(imgs_sc, gts)
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
