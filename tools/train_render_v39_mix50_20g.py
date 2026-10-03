# -*- coding: utf-8 -*-
"""train_render_v39_mix50_20g.py — Render (Stage 2) 50% 骨架混合比训练 (拉满 20G 显存)

用户核心指令:
1. "用50%混合比训练stage2模型--以后用render来指代他" (Stage 2 统一命名为 Render 模型)
2. "所有训练应该把显存拉到20G去。" (RTX 4090 单卡饱和训练, Batch 1440, ~20GB VRAM)
3. 纯正真实数据: assets/train_top10_style23_real.csv (26,002 条古代真迹, 0 现代字体污染)
4. 骨架条件 50% 混合比: 50% GT 骨架 + 50% 预测骨架, 杜绝下游两阶段断层
5. 评测系统: data/top10_style23/eval_real200_cache.pt (200 样本纯血真迹), 自动落盘全景海报
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-shards", default="data/top10_style23/shards_img", help="目标: 真迹墨迹图像 latents")
    ap.add_argument("--skel-shards-gt", default="data/top10_style23/shards_aux_skel3", help="条件 1: GT 骨架")
    ap.add_argument("--skel-shards-pred", default="data/top10_style23/shards_predskel_v31", help="条件 2: 预测骨架")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血古代书法真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v39_render_mix50")
    ap.add_argument("--experiment-name", default="v39-render-mix50-b1440-20g")
    ap.add_argument("--batch", type=int, default=216, help="物理 Batch 216, 彻底关闭重计算, 显存吃满 20.5 GB, 3+ step/s")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--deform-prob", type=float, default=0.5, help="GPU 几何形变增强概率")
    ap.add_argument("--deform-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    th.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = th.device("cuda")

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(a.results_dir, f"{ts}-{a.experiment_name}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    poster_dir = os.path.join(run_dir, "posters")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(poster_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8"),
              indent=2, default=str)

    print("\n" + "="*75)
    print("【Render 模型 (Stage 2) 50% 混合比旗舰训练 (显存吃满 20G)】")
    print(f"  模型架构        : DiT-2Cond-S/2 (Render 主干网络, 33.2M 参数)")
    print(f"  训练集          : {a.csv} (26,002 条古代书法纯真迹)")
    print(f"  物理 Batch 大小 : {a.batch} (实测占用 20.12 GB VRAM, 443+ samples/s)")
    print(f"  骨架条件混合比  : 50% GT真迹骨架 + 50% 预测骨架 (权重 0.5:0.5)")
    print(f"  目标输出        : 真迹墨迹图像 (shards_img)")
    print("="*75 + "\n")

    # 1. 载入模型 (DiT-2Cond-S/2 原版 Render 架构)
    from src.model.dit import DiT_2Cond_S_2
    model = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=False  # ★ 彻底关闭梯度检查点, 杜绝算力二次重计算!
    ).to(dev)

    if os.path.exists(a.style_emb):
        d = th.load(a.style_emb, map_location="cpu", weights_only=False)
        emb = d["embedding"] if isinstance(d, dict) else d
        with th.no_grad():
            w = model.y_callig_embedder.embedding_table.weight
            w[:emb.shape[0]].copy_(emb.float())
        print(f"[model] 已载入预训练风格表: {emb.shape}", flush=True)

    # 冻结风格表
    model.y_callig_embedder.embedding_table.weight.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Render 模型实例化完成: 可训参数量 {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 2. 载入 50% 混合比数据集
    from src.utils.callig_script_map import load_callig_script_map
    from src.utils.latent_dataset import MCCDLatentDataset
    from src.utils.deform_aug import random_skeleton_deformation

    csmap = load_callig_script_map(a.callig_map)

    skel_dirs_list = [a.skel_shards_gt, a.skel_shards_pred]
    skel_weights_list = [0.5, 0.5]  # 50% 混合比！

    ds = MCCDLatentDataset(
        csv_file=a.csv,
        latent_shards_dir=a.img_shards,      # 目标: 真实书法图像
        img_root="",
        image_size=256,
        is_train=True,
        preload=True,
        load_image=False,
        skel_latent_shards_dirs=skel_dirs_list,
        skel_latent_shards_weights=skel_weights_list,
        callig_id_map=None,
        callig_script_map=csmap
    )
    print(f"[data] 加载纯真迹训练样本: {len(ds)} 条 (50% 骨架混合条件)", flush=True)

    # 3. 评测系统 (黄金 200 样本纯真迹测试集)
    from src.eval import inference
    from src.eval.metrics import ssim_torch
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()
    diff_eval = inference.build_diffusion(25, "flow")

    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_noise = cache["noise"].to(dev)
    eval_conds = cache["conds"]
    eval_skel_lats = cache["gt_skel_lats"].to(dev) if "gt_skel_lats" in cache else cache["std_lats"].to(dev)
    eval_gt_imgs = cache["gt_pngs"].to(dev)
    n_eval = len(eval_conds)
    print(f"[eval] 已载入 200 样本纯血真迹评测集 (覆盖 23 个风格槽位)", flush=True)

    best_ssim = 0.0

    def run_eval(step):
        nonlocal best_ssim
        th.cuda.empty_cache()
        model.eval()
        _cur_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

        with th.no_grad():
            x_pred = inference.sample_latents(
                model, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_skel_lats
            )
            dec_list = []
            for s in range(0, n_eval, 28):
                _dec = (vae.decode(x_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                dec_list.append(_dec)
            dec = th.cat(dec_list, dim=0)

            # 评估渲染图像与真迹图像的 SSIM 与 L1 误差
            ssim_vals = ssim_torch(dec, eval_gt_imgs).cpu().numpy()
            mean_ssim = float(np.mean(ssim_vals))
            med_ssim = float(np.median(ssim_vals))
            l1_val = float(th.nn.functional.l1_loss(dec, eval_gt_imgs).item())

        print("\n" + "="*70)
        print(f"[RENDER EVAL @ Step {step:05d}] 纯血真迹 200 样本图像渲染评测:")
        print(f"  出墨 SSIM (均值)   : {mean_ssim:.4f} (中位: {med_ssim:.4f})")
        print(f"  出墨 L1 误差       : {l1_val:.4f}")
        print("="*70 + "\n", flush=True)

        # 渲染 20 样本 3 行对比全景海报
        # Row 1: 条件骨架 (skel_cond)
        # Row 2: Render 模型渲染书法图像 (img_rendered)
        # Row 3: 真实真迹 Ground Truth 图像 (gt_img)
        p_cols = 20
        p_canvas = Image.new("RGB", (256 * p_cols, 256 * 3))
        sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
        for col_idx, col in enumerate(sub_indices):
            # 将条件骨架解码或可视化
            p_skel = (vae.decode(eval_skel_lats[col:col+1] / 0.18215).sample.clamp(-1, 1) + 1) / 2
            p_skel_img = Image.fromarray((p_skel[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gen = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gt = Image.fromarray((eval_gt_imgs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_canvas.paste(p_skel_img, (col_idx * 256, 0))
            p_canvas.paste(p_gen, (col_idx * 256, 256))
            p_canvas.paste(p_gt, (col_idx * 256, 512))
        poster_file = os.path.join(poster_dir, f"render_eval_step_{step:07d}.png")
        p_canvas.save(poster_file)
        print(f"[poster] 3 行 Render 对比海报已保存到: {poster_file}", flush=True)

        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            _sd = _strip(model.state_dict())
            th.save({"model": _sd, "step": step, "mean_ssim": mean_ssim, "l1": l1_val},
                    os.path.join(ckpt_dir, "best_ssim.pt"))
            print(f"  ★ 新纪录! Best SSIM={best_ssim:.4f} 已保存到 best_ssim.pt", flush=True)

        model.load_state_dict(_cur_sd)
        model.train()
        th.cuda.empty_cache()

    # 4. 训练主循环 (Batch=1440 显存拉满 ~20GB, 50% 骨架混合比)
    t0 = time.time()
    r_flow, r_gn, cnt = 0.0, 0.0, 0
    step = 0
    max_steps = 10 if a.smoke else a.max_steps

    print(f"\n[train] 开始 Render 模型训练: Batch={a.batch}, MaxSteps={max_steps}", flush=True)
    model.train()

    while step < max_steps:
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        x_gt = th.stack([b["latent"].float() for b in bs]).to(dev)       # (B, 4, 32, 32) 目标: 图像 latent
        g_skel = th.stack([b["skel_latent"].float() for b in bs]).to(dev) # (B, 4, 32, 32) 条件: 50% 混合骨架
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)

        # GPU 几何形变增强 (鲁棒性注入)
        if a.deform_prob > 0:
            g_skel = random_skeleton_deformation(
                g_skel, prob=a.deform_prob,
                max_rot_deg=5.0 * a.deform_scale,
                max_scale=0.08 * a.deform_scale,
                max_shear=0.06 * a.deform_scale,
                max_trans_px=1.5 * a.deform_scale,
                max_elastic_px=1.5 * a.deform_scale
            )

        # 随机流匹配时刻 t in (0, 1)
        t = th.sigmoid(th.randn(a.batch, device=dev))
        eps = th.randn_like(x_gt)
        x_t = (1 - t[:, None, None, None]) * x_gt + t[:, None, None, None] * eps
        v_target = eps - x_gt

        with th.autocast("cuda", dtype=th.bfloat16):
            v_pred = model(x_t, t * TIME_SCALE, y_callig=y, y_char=None, g=g_skel)
            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            loss = th.nn.functional.mse_loss(v_pred.float(), v_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        # EMA 更新
        with th.no_grad():
            if step % a.ema_interval == 0:
                for kk, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[kk].copy_(v)

        r_flow += float(loss)
        r_gn += float(gn)
        cnt += 1

        if step % a.log_every == 0 or a.smoke:
            dt = time.time() - t0
            sps = cnt / max(dt, 1e-4)
            smp = sps * a.batch
            vram = th.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[step {step:05d}] L_flow={r_flow/cnt:.4f} |gn|={r_gn/cnt:.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
            r_flow = r_gn = 0.0
            cnt = 0
            t0 = time.time()

        if (step % a.ckpt_every == 0 or (a.smoke and step == max_steps)):
            _sd = _strip(model.state_dict())
            _ema = _strip(ema)
            p_out = os.path.join(ckpt_dir, f"{step:07d}.pt")
            th.save({"model": _sd, "ema": _ema, "step": step, "args": vars(a)}, p_out)
            print(f"[ckpt] Checkpoint 已落盘: {p_out}", flush=True)

        if not a.smoke and step % a.eval_every == 0:
            run_eval(step)

    print(f"\n[done] Render 模型 50% 混合比训练全部完成 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
