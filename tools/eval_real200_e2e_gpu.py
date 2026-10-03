#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_real200_e2e_gpu.py — 在 GPU 上以正统两阶段流水线评估 200 条纯血真迹测试集端到端成绩"""
import os, sys, json, csv, re, time, argparse, math
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

dev = th.device("cuda" if th.cuda.is_available() else "cpu")

from src.eval import model_io
from src.eval.in_mem_eval import _get_vae
from src.eval.metrics import ssim_torch, frag_ratio
from src.eval.inference import sample_latents, build_diffusion


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default="exp/v37_skelnet_sp/20261002-123800-v37-sp-real26k/checkpoints/0005000.pt")
    ap.add_argument("--bak-ckpt", default="exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--gen-steps", type=int, default=25)
    ap.add_argument("--bak-steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out-dir", default="exp/v37_skelnet_sp/eval_real200_e2e")
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"\n=================================================================")
    print(f"【两阶段端到端真迹评测系统 (200 样本纯血真迹)】")
    print(f"  Stage 1 SkelNet-Sp : {a.gen_ckpt}")
    print(f"  Stage 2 Backbone   : {a.bak_ckpt}")
    print(f"=================================================================\n")

    # 1. 载入模型
    from src.model.dit import DiT_2Cond_Sp_2, DiT_2Cond_S_2
    
    # 1.1 Stage 1 SkelNet-Sp
    gen = DiT_2Cond_Sp_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0
    ).to(dev).eval()

    d_gen = th.load(a.gen_ckpt, map_location="cpu", weights_only=False)
    sd_gen = d_gen.get("model", d_gen)
    sd_gen = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_gen.items()}
    gen.load_state_dict(sd_gen)
    print(f"[model] Stage 1 SkelNet-Sp (65.4M) 已成功载入 online 权重")

    # 1.2 Stage 2 Backbone
    from src.eval import model_io
    bak, _ = model_io.load_model_from_ckpt(a.bak_ckpt, device=dev, use_ema=True)
    bak.eval()
    print(f"[model] Stage 2 Backbone 已成功载入权重", flush=True)

    # 2. 载入 200 黄金真迹评测缓存
    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    rows = cache["rows"]
    noise = cache["noise"].to(dev)
    conds = cache["conds"]
    std_lats = cache["std_lats"].to(dev)
    gt_skel_lats = cache["gt_lats"].to(dev)
    gt_skel_pngs = cache["gt_pngs"].to(dev)
    std_pngs = cache["std_pngs"].to(dev)
    n = len(rows)
    print(f"[data] 已载入 200 样本纯血真迹评测集 (覆盖 23 个风格槽位)")

    # 3. 载入真实真迹原图 (用于最终成画图像指标评测)
    vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
    diff_skel = build_diffusion(a.gen_steps, "flow")
    diff_img = build_diffusion(a.bak_steps, "flow")

    # 载入真迹原图
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    gt_imgs = []
    for r in rows:
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        gt_imgs.append(tf(Image.open(p).convert("RGB")))
    gt_imgs = th.stack(gt_imgs).to(dev)
    gt_imgs_norm = (gt_imgs + 1.0) / 2.0

    # 4. 推理
    print(f"\n[run] 开始 GPU 两阶段推理 (Batch={a.batch}):")
    t0 = time.time()
    with th.no_grad():
        # 4.1 Stage 1 生成预测骨架
        print(f"  [4.1] Stage 1 生成预测骨架 (ODE {a.gen_steps} 步)...", flush=True)
        g_pred = sample_latents(gen, diff_skel, noise, conds, cfg_scale=1.0, batch=a.batch, device=dev, skel=std_lats)

        # 4.2 Stage 2 端到端成画 (用预测骨架渲染真迹)
        print(f"  [4.2] Stage 2 端到端成画 (ODE {a.bak_steps} 步, 使用预测骨架)...", flush=True)
        x_pred_e2e = sample_latents(bak, diff_img, noise, conds, cfg_scale=1.0, batch=a.batch, device=dev, skel=g_pred.to(dev))

        # 4.3 Stage 2 Oracle 上界成画 (用真实 GT 骨架渲染, 作为理论上界对照)
        print(f"  [4.3] Stage 2 Oracle 上界成画 (ODE {a.bak_steps} 步, 使用 GT 真迹骨架)...", flush=True)
        x_pred_oracle = sample_latents(bak, diff_img, noise, conds, cfg_scale=1.0, batch=a.batch, device=dev, skel=gt_skel_lats)
    
    dt = time.time() - t0
    print(f"[run] 推理完成! 200 样本总耗时: {dt:.2f}s ({n/dt:.1f} samples/s)")

    # 5. VAE 解码与指标计算
    print(f"\n[metric] VAE 解码与多维指标计算...")
    with th.no_grad():
        dec_skel_list = []
        dec_e2e_list = []
        dec_oracle_list = []
        for s in range(0, n, 28):
            _ds = (vae.decode(g_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
            _de = (vae.decode(x_pred_e2e[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
            _do = (vae.decode(x_pred_oracle[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
            dec_skel_list.append(_ds)
            dec_e2e_list.append(_de)
            dec_oracle_list.append(_do)
        dec_skel = th.cat(dec_skel_list, dim=0)
        dec_e2e = th.cat(dec_e2e_list, dim=0)
        dec_oracle = th.cat(dec_oracle_list, dim=0)

        # ── 骨架质量指标 ──
        skel_ssim = ssim_torch(dec_skel, gt_skel_pngs).cpu().numpy()
        skel_mean_ssim = float(np.mean(skel_ssim))
        skel_med_ssim = float(np.median(skel_ssim))

        # 连通块破碎度
        pred_gray = dec_skel.mean(dim=1).cpu().numpy()
        gt_gray = gt_skel_pngs.mean(dim=1).cpu().numpy()
        frags = [frag_ratio(pred_gray[i:i+1], gt_gray[i:i+1]) for i in range(len(pred_gray))]
        mean_frag = float(np.mean(frags))
        med_frag = float(np.median(frags))

        # ── 最终图像端到端指标 (E2E) ──
        e2e_ssim = ssim_torch(dec_e2e, gt_imgs_norm).cpu().numpy()
        e2e_mean_ssim = float(np.mean(e2e_ssim))
        e2e_med_ssim = float(np.median(e2e_ssim))

        e2e_gray = dec_e2e.mean(dim=1)
        gt_img_gray = gt_imgs_norm.mean(dim=1)
        e2e_mask = (e2e_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
        gt_mask = (gt_img_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
        e2e_ink_ssim = float(np.mean(ssim_torch(e2e_mask, gt_mask).cpu().numpy()))

        # ── 图像 Oracle 上界指标 ──
        ora_ssim = ssim_torch(dec_oracle, gt_imgs_norm).cpu().numpy()
        ora_mean_ssim = float(np.mean(ora_ssim))
        ora_med_ssim = float(np.median(ora_ssim))

    print("\n" + "="*70)
    print("【两阶段纯血真迹 200 样本端到端评测最终成绩单】:")
    print("-" * 70)
    print(f"  1. Stage 1 骨架生成质量:")
    print(f"     - 骨架 SSIM (均值)   : {skel_mean_ssim:.4f} (中位: {skel_med_ssim:.4f})")
    print(f"     - 破碎度 (frag_ratio): {mean_frag:.3f} (中位: {med_frag:.3f}) [1.0=完全连贯无破碎!]")
    print(f"  2. Stage 2 两阶段端到端 (E2E):")
    print(f"     - 成画 SSIM (均值)   : {e2e_mean_ssim:.4f} (中位: {e2e_med_ssim:.4f})")
    print(f"     - 墨迹 SSIM (Ink)    : {e2e_ink_ssim:.4f}")
    print(f"  3. Stage 2 Oracle 理论上界 (真实 GT 骨架输入):")
    print(f"     - 上界 SSIM (均值)   : {ora_mean_ssim:.4f} (中位: {ora_med_ssim:.4f})")
    print("="*70 + "\n")

    # 6. 渲染 20 样本 4 行对比全景海报
    # Row 1: 标准字输入骨架 (g_std)
    # Row 2: Stage 1 生成骨架 (g_pred)
    # Row 3: Stage 2 端到端成画 (x_e2e)
    # Row 4: 历史真迹 Ground Truth (x_gt)
    p_cols = 20
    p_canvas = Image.new("RGB", (256 * p_cols, 256 * 4))
    sub_indices = np.linspace(0, n - 1, p_cols, dtype=int)

    for col_idx, col in enumerate(sub_indices):
        p_std = Image.fromarray((std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_skel = Image.fromarray((dec_skel[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_e2e = Image.fromarray((dec_e2e[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        p_gt = Image.fromarray((gt_imgs_norm[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        p_canvas.paste(p_std, (col_idx * 256, 0))
        p_canvas.paste(p_skel, (col_idx * 256, 256))
        p_canvas.paste(p_e2e, (col_idx * 256, 512))
        p_canvas.paste(p_gt, (col_idx * 256, 768))

    poster_path = os.path.join(a.out_dir, "real200_e2e_4row_poster.png")
    p_canvas.save(poster_path)
    print(f"[poster] 4 行全景对比海报已保存到: {poster_path}")

    # 保存量化指标 JSON
    metrics_res = {
        "skel_mean_ssim": skel_mean_ssim,
        "skel_med_ssim": skel_med_ssim,
        "skel_mean_frag": mean_frag,
        "skel_med_frag": med_frag,
        "e2e_mean_ssim": e2e_mean_ssim,
        "e2e_med_ssim": e2e_med_ssim,
        "e2e_ink_ssim": e2e_ink_ssim,
        "oracle_mean_ssim": ora_mean_ssim,
        "oracle_med_ssim": ora_med_ssim,
        "n_samples": n,
        "elapsed_sec": dt
    }
    with open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
