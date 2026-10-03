# -*- coding: utf-8 -*-
"""train_render_cascaded_curriculum_20g.py — Render 模型 50% -> 60% -> 70% -> 80% -> 90% 级联渐进式训练系统

核心技术实现:
1. 底座继承: 显式从 50% 混合黄金存档 (exp/v39_render_mix50/.../best_ssim.pt) 起步 resume。
2. 级联课程 (Cascaded Curriculum):
   - Stage 1 (Step 0 ~ 1,000)   : 60% Pred 骨架 + 40% GT 骨架 (权重 [0.6, 0.4])
   - Stage 2 (Step 1,000 ~ 2,000): 70% Pred 骨架 + 30% GT 骨架 (权重 [0.7, 0.3])
   - Stage 3 (Step 2,000 ~ 3,000): 80% Pred 骨架 + 20% GT 骨架 (权重 [0.8, 0.2])
   - Stage 4 (Step 3,000 ~ 4,500): 90% Pred 骨架 + 10% GT 骨架 (权重 [0.9, 0.1])
3. 评测系统: 严格执行“所有 eval 都用 pred”，100% 采用 SkelNet 预测骨架测试，无任何 GT 骨架泄露。
4. 显存与算力: Batch 208, 显存吃满 20.44 GB VRAM, 关闭梯度重计算 (2.7+ step/s, 560+ samples/s)。
5. 自动在每个阶段末尾落盘专属 Checkpoint 与 4 行端到端对比全景海报。
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
    ap.add_argument("--img-shards", default="data/top10_style23/shards_img")
    ap.add_argument("--skel-shards-pred", default="data/top10_style23/shards_predskel_v31")
    ap.add_argument("--skel-shards-gt", default="data/top10_style23/shards_aux_skel3")
    ap.add_argument("--init-ckpt", default="exp/v39_render_mix50/20261002-200307-v39-render-mix50-b208-20g/checkpoints/best_ssim.pt", help="50% 混合底座")
    ap.add_argument("--skelnet-ckpt", default="exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v42_render_curriculum")
    ap.add_argument("--experiment-name", default="v42-render-curriculum-b208-20g")
    ap.add_argument("--batch", type=int, default=208, help="物理 Batch 208, 显存吃满 20.44 GB, ~2.7 step/s")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--deform-prob", type=float, default=0.5)
    ap.add_argument("--deform-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# 级联阶段配置: (阶段名, Pred 骨架权重, GT 骨架权重, 持续步数)
CURRICULUM_STAGES = [
    ("Stage 1 (60% Pred)", 0.6, 0.4, 1000),
    ("Stage 2 (70% Pred)", 0.7, 0.3, 1000),
    ("Stage 3 (80% Pred)", 0.8, 0.2, 1000),
    ("Stage 4 (90% Pred)", 0.9, 0.1, 1500),
]


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
    print("【Render 模型 50% -> 60% -> 70% -> 80% -> 90% 级联渐进式重训系统】")
    print(f"  初始底座        : {a.init_ckpt} (50% 混合最佳权重)")
    print(f"  训练集          : {a.csv} (26,002 条古代书法纯真迹)")
    print(f"  物理 Batch 大小 : {a.batch} (显存饱和在 20.44 GB VRAM, ~2.7 step/s)")
    print(f"  评测机制        : ★ 100% 采用 SkelNet 预测骨架评测, 杜绝 GT 泄露")
    print("  级联梯度规划    :")
    total_steps = 0
    for name, pw, gw, n_steps in CURRICULUM_STAGES:
        print(f"    - {name:20s}: 权重 [{pw:.1f}, {gw:.1f}], 计划步数 {n_steps:4d} 步")
        total_steps += n_steps
    print(f"  总训练规划步数  : {total_steps} 步 (预计总用时 ~27 分钟)")
    print("="*75 + "\n")

    # 1. 载入模型 (关闭梯度重计算)
    from src.model.dit import DiT_2Cond_S_2
    render = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=False
    ).to(dev)

    if os.path.exists(a.init_ckpt):
        d_ren = th.load(a.init_ckpt, map_location="cpu", weights_only=False)
        sd_ren = d_ren.get("model", d_ren)
        sd_ren = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_ren.items()}
        render.load_state_dict(sd_ren)
        print(f"[model] Render 模型成功继承 50% 混合黄金底座 -> {a.init_ckpt}", flush=True)

    render.y_callig_embedder.embedding_table.weight.requires_grad_(False)
    n_params = sum(p.numel() for p in render.parameters() if p.requires_grad)
    print(f"[model] Render 可训参数量: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    # 载入 SkelNet 模型专用于在线生成 100% Pred 评测骨架
    skelnet = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=False
    ).to(dev).eval()

    if os.path.exists(a.skelnet_ckpt):
        d_skel = th.load(a.skelnet_ckpt, map_location="cpu", weights_only=False)
        sd_skel = d_skel.get("model", d_skel)
        sd_skel = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_skel.items()}
        skelnet.load_state_dict(sd_skel)
        print(f"[model] SkelNet 黄金基模已就绪, 专供 100% Pred 评测生成 -> {a.skelnet_ckpt}", flush=True)

    for p in skelnet.parameters():
        p.requires_grad_(False)

    opt = th.optim.AdamW([p for p in render.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=a.lr * 0.1)
    ema = {k: v.detach().clone() for k, v in render.state_dict().items()}

    # 2. 载入数据集
    from src.utils.callig_script_map import load_callig_script_map
    from src.utils.latent_dataset import MCCDLatentDataset
    from src.utils.deform_aug import random_skeleton_deformation

    csmap = load_callig_script_map(a.callig_map)

    skel_dirs_list = [a.skel_shards_pred, a.skel_shards_gt]
    init_weights = [0.6, 0.4]

    ds = MCCDLatentDataset(
        csv_file=a.csv,
        latent_shards_dir=a.img_shards,      # 目标: 真实书法图像
        img_root="",
        image_size=256,
        is_train=True,
        preload=True,
        load_image=False,
        skel_latent_shards_dirs=skel_dirs_list,
        skel_latent_shards_weights=init_weights,
        callig_id_map=None,
        callig_script_map=csmap
    )
    print(f"[data] 纯真迹数据集加载完成: {len(ds)} 条 (动态骨架混合)", flush=True)

    # 3. 评测系统 (100% 采用 Pred 骨架评测)
    from src.eval import inference
    from src.eval.metrics import ssim_torch
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()
    diff_eval = inference.build_diffusion(25, "flow")

    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_rows = cache["rows"]
    eval_noise = cache["noise"].to(dev)
    eval_conds = cache["conds"]
    eval_std_lats = cache["std_lats"].to(dev)
    eval_std_pngs = cache["std_pngs"].to(dev)
    n_eval = len(eval_rows)

    # 载入 200 样本历史碑帖真迹原图
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    eval_gt_imgs = []
    for r in eval_rows:
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        eval_gt_imgs.append(tf(Image.open(p).convert("RGB")))
    eval_gt_imgs = th.stack(eval_gt_imgs).to(dev)
    eval_gt_imgs_norm = (eval_gt_imgs + 1.0) / 2.0

    print("[eval] 正在为 200 评测样本生成 100% Pred 骨架条件 (绝不使用 GT 骨架)...", flush=True)
    with th.no_grad():
        eval_pred_skel_lats = inference.sample_latents(
            skelnet, diff_eval, eval_noise, eval_conds,
            cfg_scale=1.0, batch=50, device=dev, skel=eval_std_lats
        )
        dec_pred_skels = []
        for s in range(0, n_eval, 28):
            _dec = (vae.decode(eval_pred_skel_lats[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
            dec_pred_skels.append(_dec)
        eval_pred_skel_pngs = th.cat(dec_pred_skels, dim=0)
    print(f"[eval] 100% 预测骨架条件构建完成: {eval_pred_skel_lats.shape}", flush=True)

    best_ssim = 0.0

    def run_eval(step, stage_label):
        nonlocal best_ssim
        th.cuda.empty_cache()
        render.eval()
        _cur_sd = {k: v.detach().clone() for k, v in render.state_dict().items()}

        with th.no_grad():
            x_pred = inference.sample_latents(
                render, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_pred_skel_lats
            )
            dec_list = []
            for s in range(0, n_eval, 28):
                _dec = (vae.decode(x_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                dec_list.append(_dec)
            dec = th.cat(dec_list, dim=0)

            ssim_vals = ssim_torch(dec, eval_gt_imgs_norm).cpu().numpy()
            mean_ssim = float(np.mean(ssim_vals))
            med_ssim = float(np.median(ssim_vals))
            l1_val = float(th.nn.functional.l1_loss(dec, eval_gt_imgs_norm).item())

        print("\n" + "="*70)
        print(f"[{stage_label} @ Step {step:05d}] 纯血真迹 100% Pred 骨架出墨评测:")
        print(f"  出墨 SSIM (基于 Pred 骨架, 均值): {mean_ssim:.4f} (中位: {med_ssim:.4f})")
        print(f"  出墨 L1 误差 (基于 Pred 骨架)   : {l1_val:.4f}")
        print("="*70 + "\n", flush=True)

        p_cols = 20
        p_canvas = Image.new("RGB", (256 * p_cols, 256 * 4))
        sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
        for col_idx, col in enumerate(sub_indices):
            p_std = Image.fromarray((eval_std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_pred_sk = Image.fromarray((eval_pred_skel_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_ren = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gt = Image.fromarray((eval_gt_imgs_norm[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_canvas.paste(p_std, (col_idx * 256, 0))
            p_canvas.paste(p_pred_sk, (col_idx * 256, 256))
            p_canvas.paste(p_ren, (col_idx * 256, 512))
            p_canvas.paste(p_gt, (col_idx * 256, 768))

        poster_file = os.path.join(poster_dir, f"curriculum_eval_step_{step:07d}.png")
        p_canvas.save(poster_file)
        print(f"[poster] 4 行全景对比海报已保存到: {poster_file}", flush=True)

        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            _sd = _strip(render.state_dict())
            th.save({"model": _sd, "step": step, "mean_ssim": mean_ssim, "l1": l1_val},
                    os.path.join(ckpt_dir, "best_ssim.pt"))
            print(f"  ★ 新纪录! Best SSIM={best_ssim:.4f} 已保存到 best_ssim.pt", flush=True)

        render.load_state_dict(_cur_sd)
        render.train()
        th.cuda.empty_cache()

    # 4. 级联训练主循环
    t0 = time.time()
    r_flow, r_gn, cnt = 0.0, 0.0, 0
    current_global_step = 0
    render.train()

    for stage_idx, (stage_name, pred_w, gt_w, n_steps) in enumerate(CURRICULUM_STAGES):
        # 动态更新数据集采样权重
        ds.skel_latent_shards_weights = [pred_w, gt_w]
        print(f"\n{'#'*75}")
        print(f"【进入级联新阶段】 {stage_name} (Pred={pred_w*100:.0f}%, GT={gt_w*100:.0f}%), 计划执行 {n_steps} 步")
        print(f"{'#'*75}\n", flush=True)

        steps_this_stage = 0
        target_steps = 5 if a.smoke else n_steps

        while steps_this_stage < target_steps:
            k = np.random.randint(0, len(ds), a.batch)
            bs = [ds[int(j)] for j in k]
            x_gt = th.stack([b["latent"].float() for b in bs]).to(dev)
            g_skel = th.stack([b["skel_latent"].float() for b in bs]).to(dev)
            y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)

            if a.deform_prob > 0:
                g_skel = random_skeleton_deformation(
                    g_skel, prob=a.deform_prob,
                    max_rot_deg=5.0 * a.deform_scale,
                    max_scale=0.08 * a.deform_scale,
                    max_shear=0.06 * a.deform_scale,
                    max_trans_px=1.5 * a.deform_scale,
                    max_elastic_px=1.5 * a.deform_scale
                )

            t = th.sigmoid(th.randn(a.batch, device=dev))
            eps = th.randn_like(x_gt)
            x_t = (1 - t[:, None, None, None]) * x_gt + t[:, None, None, None] * eps
            v_target = eps - x_gt

            with th.autocast("cuda", dtype=th.bfloat16):
                v_pred = render(x_t, t * TIME_SCALE, y_callig=y, y_char=None, g=g_skel)
                if isinstance(v_pred, tuple):
                    v_pred = v_pred[0]
                loss = th.nn.functional.mse_loss(v_pred.float(), v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = th.nn.utils.clip_grad_norm_(render.parameters(), 1.0)
            opt.step()
            sched.step()

            steps_this_stage += 1
            current_global_step += 1

            with th.no_grad():
                if current_global_step % a.ema_interval == 0:
                    for kk, v in render.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                        else:
                            ema[kk].copy_(v)

            r_flow += float(loss)
            r_gn += float(gn)
            cnt += 1

            if current_global_step % a.log_every == 0 or a.smoke:
                dt = time.time() - t0
                sps = cnt / max(dt, 1e-4)
                smp = sps * a.batch
                vram = th.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[{stage_name} | Step {current_global_step:05d}] L_flow={r_flow/cnt:.4f} |gn|={r_gn/cnt:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
                r_flow = r_gn = 0.0
                cnt = 0
                t0 = time.time()

        # 阶段完成：专属 Checkpoint 存储与评测
        _sd = _strip(render.state_dict())
        p_stage_ckpt = os.path.join(ckpt_dir, f"ckpt_{int(pred_w*100)}pct_pred_step{current_global_step:05d}.pt")
        th.save({"model": _sd, "step": current_global_step, "pred_w": pred_w, "gt_w": gt_w}, p_stage_ckpt)
        print(f"\n[stage-done] ★ {stage_name} 顺利完成! Checkpoint: {p_stage_ckpt}", flush=True)

        if not a.smoke:
            run_eval(current_global_step, stage_name)

    print(f"\n[done] 级联渐进式训练全部阶段圆满收官 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
