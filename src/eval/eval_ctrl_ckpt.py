# -*- coding: utf-8 -*-
"""
eval_ctrl_ckpt.py — 独立 ControlNet 评测（GPU 采样 + 全图落盘 + CPU 指标）。

用法示例:
    python src/eval/eval_ctrl_ckpt.py \\
        --main-ckpt 5script/results/s30_dino_char_strong_pretrain/.../0132500.pt \\
        --ctrl-ckpt 5script/results/s31_ctrl_gt_skel_1px/.../0042500.pt \\
        --eval-csv 5script/eval_fame_strict_clean.csv \\
        --skel-latent-dir final_skel_latents_fame_1px \\
        --skel-root final_skel1_fame --img-root final_imgs_256 \\
        --out-dir 5script/results/s31_ctrl_gt_skel_1px/manual_eval/step0042500 \\
        --n 100 --cfg 0.7 --steps 50 --device cuda

行为:
  * 加载主模型 + ctrl encoder（从 ctrl-ckpt 的 ema/ctrl 键，自动剥 _orig_mod. 前缀）
  * base（无 skel）与 ctrl（GT skel）各采样一次（flow, heun, cfg）
  * 所有图片落盘: {out}/base/base{i}.png, {out}/ctrl/ctrl{i}.png,
    {out}/ctrl/gt{i}.png, {out}/ctrl/skel{i}.png（skel latent 会 VAE decode 成图）
  * 指标: MSE/SSIM/skel_iou/LPIPS → {out}/metrics.json
"""
import os
import sys
import json
import argparse
import traceback

import numpy as np
import torch

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.model.controlnet import ControlNetDiT, load_main_model
from src.eval.inference import (
    make_eval_cache, sample_latents, decode_and_save,
    compute_metrics, load_eval_vae, build_diffusion,
)


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [eval-ckpt] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Standalone ControlNet eval (GPU sample + full-disk save)")
    # 模型 ckpt
    ap.add_argument("--main-ckpt", required=True, help="主模型 ckpt (S21/S30 base)")
    ap.add_argument("--ctrl-ckpt", default="", help="ctrl ckpt; 空=纯 base 评测")
    # 数据
    ap.add_argument("--eval-csv", required=True, help="eval 集 csv (有无 skel latent 均可)")
    ap.add_argument("--skel-latent-dir", default="", help="skel VAE latent 目录 (优先); 空=用 PNG skel_root")
    ap.add_argument("--skel-root", default="final_skel1_fame", help="skel PNG 根 (fallback)")
    ap.add_argument("--img-root", default="final_imgs_256", help="GT 图根")
    # eval 协议
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dit-batch", type=int, default=8)
    ap.add_argument("--vae-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # 输出
    ap.add_argument("--out-dir", required=True, help="落盘目录; base/ctrl 子目录自动建")
    # 环境
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--vae-path", default="pretrained_models/sd-vae-ft-ema")
    return ap.parse_args()


def build_model(main_ckpt, ctrl_ckpt, device):
    """构建 ControlNetDiT: 主模型加载 + ctrl encoder 加载。返回 (model, main_missing).
    若 ctrl_ckpt 内含 main.* 权重 (REPA/from-scratch 存盘), 则用其覆盖主模型
    (优先 ema_model/main 键), 使 REPA 对主模型的修改也被纳入评测。
    """
    # 主模型配置与 S30/S31 系一致（DINO 真迹字表 + mlp proj + freeze）
    main = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=main_ckpt, device="cpu",
        num_calligraphers=1013, num_characters=35130,
        condition_fusion="factorized_add",
        callig_embed_dim=128, char_embed_dim=384, char_proj_mode="mlp",
        freeze_char_table=True,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.30,
        cond_drop_which_glyph_prob=0.85,
        use_checkpoint=False, learn_sigma=None, diffusion_type="flow",
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa",
    )
    ctrl = ControlNetDiT(
        main, cond_in_channels=4, train_ctrl_only=True,
        ctrl_depth=0, ctrl_hidden=0, ctrl_num_heads=0,
        injection="modulate", null_cond="gaussian",
        norm_type="rms", mlp_type="swiglu", qk_norm=True,
        rope=True, rope_theta=100.0, attn_impl="sdpa",
    ).to(device)
    ctrl.eval()

    if ctrl_ckpt and os.path.exists(ctrl_ckpt):
        ck = torch.load(ctrl_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("ctrl") or ck

        # 1) 若 ckpt 内含 main.* (REPA/from-scratch 存盘), 覆盖主模型
        main_keys = {k: v for k, v in sd.items() if k.startswith("main.")}
        main_ema = ck.get("ema_model") or {}
        if main_ema:
            main_keys = {k: v for k, v in main_ema.items() if k.startswith("main.")}
        if main_keys:
            _m_sd = {}
            for k, v in main_keys.items():
                kk = k[len("main."):]
                kk = kk[len("_orig_mod."):] if kk.startswith("_orig_mod.") else kk
                _m_sd[kk] = v
            mm, mu = ctrl.load_state_dict({f"main.{k}": v for k, v in _m_sd.items()},
                                          strict=False)
            _log(f"main override from {ctrl_ckpt}: {len(_m_sd)} keys "
                 f"(missing={len(mm)}, unexpected={len(mu)})")

        # 2) ctrl encoder + injections (剥 _orig_mod.)
        ctrl_keys = {}
        for k, v in sd.items():
            if k.startswith("main."):
                continue
            kk = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
            ctrl_keys[kk] = v
        miss, unexp = ctrl.load_state_dict(ctrl_keys, strict=False)
        ctrl_miss = [k for k in miss if not k.startswith("main.")]
        _log(f"ctrl {ctrl_ckpt}: loaded ctrl_keys={len(ctrl_keys)} "
             f"ctrl_missing={len(ctrl_miss)} unexpected={len(unexp)}")
        if ctrl_miss:
            _log(f"  WARN ctrl missing(前8): {ctrl_miss[:8]}")
        if unexp:
            _log(f"  WARN unexpected(前8): {list(unexp)[:8]}")
    else:
        _log("no ctrl-ckpt -> 纯 base 评测 (ctrl 通道随机)")
    return ctrl


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available()
                          else "cpu")
    _log(f"device={device} n={args.n} cfg={args.cfg} steps={args.steps}")

    # 1) 模型
    ctrl = build_model(args.main_ckpt, args.ctrl_ckpt, device)

    # 2) VAE + diffusion
    vae = load_eval_vae(device, args.vae_path)
    diffusion = build_diffusion(args.steps, diffusion_type="flow",
                                flow_kwargs={"t_sampler": "logit_normal",
                                             "t_mean": 0.0, "t_std": 1.0,
                                             "flow_sampler": "heun",
                                             "heun_batch": 1, "shift": 1.0})

    # 3) eval cache (skel latent 优先; 复核覆盖率)
    skel_latent_dir = args.skel_latent_dir or None
    cache = make_eval_cache(
        args.eval_csv, args.img_root, args.skel_root, image_size=256, n=args.n,
        vae_downscale=8, latent_channels=4, scaling_factor=0.18215,
        skel_latent_shards_dir=skel_latent_dir)
    n_eff = cache["n"]
    skels_latent = cache.get("skels_latent")
    skels_png = cache.get("skels")
    if skels_latent is not None:
        nz = (skels_latent.abs().sum(dim=(1, 2, 3)) > 1e-6).sum().item()
        _log(f"skel latent: {nz}/{n_eff} 非零 (覆盖率={nz}/{n_eff})")
        skel_cond = skels_latent
    else:
        skel_cond = skels_png
        _log(f"skel PNG used (no latent dir): {n_eff}")

    # 4) base 采样 → 落盘
    out_base = os.path.join(args.out_dir, "base")
    os.makedirs(out_base, exist_ok=True)
    lat_base = sample_latents(ctrl, diffusion, cache["noise"], cache["conds"],
                              args.cfg, args.dit_batch, device, skel=None, seed=args.seed)
    n_base = decode_and_save(vae, lat_base, 0.18215, out_base, "base",
                             gts=cache["gts"], vae_batch=args.vae_batch)
    _log(f"base: {n_base} 张落盘 -> {out_base}/base{{i}}.png")

    # 5) ctrl 采样 → 落盘 (gt+skel)
    out_ctrl = os.path.join(args.out_dir, "ctrl")
    os.makedirs(out_ctrl, exist_ok=True)
    lat_ctrl = sample_latents(ctrl, diffusion, cache["noise"], cache["conds"],
                              args.cfg, args.dit_batch, device,
                              skel=skel_cond, seed=args.seed)
    n_ctrl = decode_and_save(vae, lat_ctrl, 0.18215, out_ctrl, "ctrl",
                             gts=cache["gts"], vae_batch=args.vae_batch,
                             skels=skel_cond)
    _log(f"ctrl: {n_ctrl} 张落盘 -> {out_ctrl}/ctrl{{i}}.png (+gt/skel)")

    # 6) CPU 指标
    res_base = compute_metrics(out_base, out_base, "base", n_ctrl, use_lpips=True)
    res_ctrl = compute_metrics(out_ctrl, out_ctrl, "ctrl", n_ctrl, use_lpips=True)
    result = {
        "main_ckpt": args.main_ckpt,
        "ctrl_ckpt": args.ctrl_ckpt,
        "eval_csv": args.eval_csv,
        "skel_latent_dir": args.skel_latent_dir or "(PNG)",
        "n": n_ctrl, "cfg": args.cfg, "steps": args.steps,
        "base": res_base, "ctrl": res_ctrl,
        "delta_mse": (res_ctrl.get("mse_mean", 0) - res_base.get("mse_mean", 0)),
        "delta_ssim": (res_ctrl.get("ssim_mean", 0) - res_base.get("ssim_mean", 0)),
        "delta_skel_iou": (res_ctrl.get("skel_iou_mean", 0) - res_base.get("skel_iou_mean", 0)),
    }
    if "lpips_mean" in res_base and "lpips_mean" in res_ctrl:
        result["delta_lpips"] = res_ctrl["lpips_mean"] - res_base["lpips_mean"]

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    def _fmt(r):
        return (f"mse={r.get('mse_mean', 0):.5f} ssim={r.get('ssim_mean', 0):.4f} "
                f"skel_iou={r.get('skel_iou_mean', 0):.4f} "
                f"lpips={r.get('lpips_mean', 0):.4f}")
    _log("")
    _log("==== 结果 ====")
    _log(f"base : {_fmt(res_base)}")
    _log(f"ctrl : {_fmt(res_ctrl)}")
    _log(f"ΔMSE={result['delta_mse']:+.5f}  ΔSSIM={result['delta_ssim']:+.4f}  "
         f"ΔSkelIoU={result['delta_skel_iou']:+.4f}  "
         f"ΔLPIPS={result.get('delta_lpips', float('nan')):+.4f}")
    _log(f"metrics -> {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)