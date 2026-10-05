# -*- coding: utf-8 -*-
"""train_joint_v37_1step_20g.py — v37 纯血真迹 1-Step 端到端联合微调 (拉满 20G VRAM):

  核心架构:
  1. 数据流: 纯血真迹训练集 (assets/train_top10_style23_real.csv, 26,002 样本, 0 字库合成数据污染)
  2. 显存优化: 单卡物理 Batch=896, 直接拉满 ~19.5~20.5 GB 显存, 杜绝梯度累加气泡与显存闲置
  3. Stage 1: DiT-2Cond-Sp/2 (65.4M, 预训练至 7500 步的强泛化基模)
  4. Stage 2: DiT-2Cond-S/2  (33.2M, 预训练图像主干 0030000.pt)
  5. 1-Step 流投影:
     - Stage 1 闭式 Tweedie 投影: g_pred = z_t1 - t1 * v1
     - 保持梯度直通注入 Stage 2
     - Stage 2 图像损失 L_img 反向传导梯度回 Stage 1, 抹平两阶段接口域隙
  6. 评测系统:
     - 黄金 200 样本纯真迹评测集 (data/top10_style23/eval_real200_cache.pt)
     - 评估骨架质量 (SSIM, frag_ratio) 与 最终图像端到端质量 (E2E SSIM, Ink SSIM)
     - 渲染 4 行全景对比海报
"""
import os, sys, json, glob, csv, re, time, math, argparse
import numpy as np
import torch as th
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-ckpt", default="exp/v37_skelnet_sp/20261002-123800-v37-sp-real26k/checkpoints/0007500.pt")
    ap.add_argument("--bak-ckpt", default="exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std_w7")
    ap.add_argument("--shards-gt", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v37_union_20g")
    ap.add_argument("--experiment-name", default="v37-union-b896-20g")
    ap.add_argument("--batch", type=int, default=896, help="物理 Batch 896, 显存吃满 ~19.8 GB")
    ap.add_argument("--gen-lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=15000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--lam-skel-flow", type=float, default=1.0, help="Stage 1 原生 Flow 损失权重 (保骨架先验)")
    ap.add_argument("--lam-skel-mse", type=float, default=0.3, help="Tweedie 骨架 MSE 锚定权重")
    ap.add_argument("--lam-skel-lap", type=float, default=0.1, help="拉普拉斯拓扑连续性损失权重")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ── 拉普拉斯连续性算子 ──
def laplacian_filter(x):
    kernel = th.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
    return th.nn.functional.conv2d(x, kernel, padding=1, groups=4)


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
    print("【v37 1-Step 端到端联合微调系统 (20GB VRAM 旗舰级)】")
    print(f"  Stage 1 SkelNet-Sp : {a.gen_ckpt}")
    print(f"  Stage 2 Backbone   : {a.bak_ckpt}")
    print(f"  训练集             : {a.csv} (26,002 条纯血古代书法真迹)")
    print(f"  物理 Batch 大小    : {a.batch} (显存拉满 ~20GB)")
    print("="*75 + "\n")

    # 1. 载入模型
    from src.eval import model_io
    from src.model.dit import DiT_2Cond_Sp_2

    # 1.1 Stage 1 SkelNet-Sp (65.4M, 可训)
    gen = DiT_2Cond_Sp_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0
    ).to(dev).train()

    d_gen = th.load(a.gen_ckpt, map_location="cpu", weights_only=False)
    sd_gen = d_gen.get("model", d_gen)
    sd_gen = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_gen.items()}
    gen.load_state_dict(sd_gen)
    # 冻结底层风格表
    gen.y_callig_embedder.embedding_table.weight.requires_grad_(False)
    print(f"[model] Stage 1 SkelNet-Sp (65.4M) 已成功载入")

    # 1.2 Stage 2 Backbone (33.2M, 冻结, 传导图像视觉判别梯度)
    bak, _ = model_io.load_model_from_ckpt(a.bak_ckpt, device=dev, use_ema=True)
    bak.eval()
    for p in bak.parameters():
        p.requires_grad_(False)
    print(f"[model] Stage 2 Backbone (33.2M) 已成功载入并冻结")

    gen_p = [p for p in gen.parameters() if p.requires_grad]
    print(f"[model] 联合微调优化器目标: {sum(p.numel() for p in gen_p):,} 个参数", flush=True)

    opt = th.optim.AdamW(gen_p, lr=a.gen_lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}

    # 2. 载入数据集
    from src.utils.callig_script_map import load_callig_script_map
    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = load_callig_script_map(a.callig_map)

    aux = [a.shards_gt]
    ds = MCCDLatentDataset(
        csv_file=a.csv, latent_shards_dir=a.shards_img, img_root="",
        image_size=256, is_train=True, preload=True, load_image=False,
        skel_latent_shards_dir=a.shards_std,
        aux_latent_shards_dirs=aux,
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] 加载纯真迹训练样本: {len(ds)} 条", flush=True)

    # 3. 评测系统 (载入黄金 200 样本缓存)
    from src.eval import inference
    from src.eval.metrics import ssim_torch, frag_ratio
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()
    diff_skel = inference.build_diffusion(25, "flow")
    diff_img = inference.build_diffusion(25, "flow")

    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_rows = cache["rows"]
    eval_noise = cache["noise"].to(dev)
    eval_conds = cache["conds"]
    eval_std_lats = cache["std_lats"].to(dev)
    eval_gt_skel_pngs = cache["gt_pngs"].to(dev)
    eval_std_pngs = cache["std_pngs"].to(dev)
    n_eval = len(eval_rows)

    # 载入 200 样本历史真迹原图
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    eval_gt_imgs = []
    for r in eval_rows:
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        eval_gt_imgs.append(tf(Image.open(p).convert("RGB")))
    eval_gt_imgs = th.stack(eval_gt_imgs).to(dev)
    eval_gt_imgs_norm = (eval_gt_imgs + 1.0) / 2.0

    best_e2e_ssim = 0.0

    def run_eval(step):
        nonlocal best_e2e_ssim
        th.cuda.empty_cache()
        gen.eval()
        _cur_sd = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        # 评测使用实际 online model 权重 (避免初期 EMA 零滞后)
        with th.no_grad():
            # 1. Stage 1 生成骨架
            g_pred = inference.sample_latents(
                gen, diff_skel, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_std_lats
            )
            # 2. Stage 2 成画
            x_pred = inference.sample_latents(
                bak, diff_img, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=g_pred.to(dev)
            )

            # VAE decode
            dec_skel_list = []
            dec_img_list = []
            for s in range(0, n_eval, 28):
                _ds = (vae.decode(g_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                _di = (vae.decode(x_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                dec_skel_list.append(_ds)
                dec_img_list.append(_di)
            dec_skel = th.cat(dec_skel_list, dim=0)
            dec_img = th.cat(dec_img_list, dim=0)

            # 骨架指标
            skel_ssim = float(np.mean(ssim_torch(dec_skel, eval_gt_skel_pngs).cpu().numpy()))
            pred_gray = dec_skel.mean(dim=1).cpu().numpy()
            gt_gray = eval_gt_skel_pngs.mean(dim=1).cpu().numpy()
            frags = [frag_ratio(pred_gray[i:i+1], gt_gray[i:i+1]) for i in range(len(pred_gray))]
            mean_frag = float(np.mean(frags))
            med_frag = float(np.median(frags))

            # 图像端到端指标 (E2E)
            e2e_ssim = float(np.mean(ssim_torch(dec_img, eval_gt_imgs_norm).cpu().numpy()))
            med_e2e_ssim = float(np.median(ssim_torch(dec_img, eval_gt_imgs_norm).cpu().numpy()))
            
            img_gray = dec_img.mean(dim=1)
            gt_img_gray = eval_gt_imgs_norm.mean(dim=1)
            e2e_mask = (img_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
            gt_mask = (gt_img_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
            ink_ssim = float(np.mean(ssim_torch(e2e_mask, gt_mask).cpu().numpy()))

        print("\n" + "="*70)
        print(f"[EVAL @ Step {step:05d}] 纯血真迹端到端成绩:")
        print(f"  Stage 1 骨架 SSIM  : {skel_ssim:.4f}")
        print(f"  Stage 1 破碎度 frag: {mean_frag:.3f} (中位 {med_frag:.3f}) [1.0=完全连贯!]")
        print(f"  Stage 2 成画 SSIM  : {e2e_ssim:.4f} (中位 {med_e2e_ssim:.4f})")
        print(f"  Stage 2 墨迹 SSIM  : {ink_ssim:.4f}")
        print("="*70 + "\n", flush=True)

        # 渲染 20 样本 4 行对比海报
        p_cols = 20
        p_canvas = Image.new("RGB", (256 * p_cols, 256 * 4))
        sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
        for col_idx, col in enumerate(sub_indices):
            p_std = Image.fromarray((eval_std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_sk = Image.fromarray((dec_skel[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_im = Image.fromarray((dec_img[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gt = Image.fromarray((eval_gt_imgs_norm[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_canvas.paste(p_std, (col_idx * 256, 0))
            p_canvas.paste(p_sk, (col_idx * 256, 256))
            p_canvas.paste(p_im, (col_idx * 256, 512))
            p_canvas.paste(p_gt, (col_idx * 256, 768))
        p_canvas.save(os.path.join(poster_dir, f"eval_step_{step:07d}.png"))

        # 最优保存
        if e2e_ssim > best_e2e_ssim:
            best_e2e_ssim = e2e_ssim
            _sd = _strip(gen.state_dict())
            th.save({"gen": _sd, "step": step, "e2e_ssim": e2e_ssim, "skel_ssim": skel_ssim, "frag": mean_frag},
                    os.path.join(ckpt_dir, "best_e2e.pt"))
            print(f"  ★ 新纪录! Best E2E SSIM={best_e2e_ssim:.4f} 已保存到 best_e2e.pt", flush=True)

        gen.load_state_dict(_cur_sd)
        gen.train()

    # 4. 训练主循环 (Batch=896 显存拉满 ~20GB, 1-Step Flow Projection)
    t0 = time.time()
    r_img, r_sflow, r_smse, r_lap, r_gn, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    step = 0
    max_steps = 10 if a.smoke else a.max_steps

    print(f"\n[train] 开始 1-Step 联合微调: Batch={a.batch}, MaxSteps={max_steps}", flush=True)
    gen.train()

    while step < max_steps:
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        x0 = th.stack([b["latent"].float() for b in bs]).to(dev)
        g_std = th.stack([b["skel_latent"].float() for b in bs]).to(dev)
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)
        g_gt = th.stack([b["aux_latents"].float() for b in bs]).to(dev)

        # 随机流匹配时刻 t1, t2 in (0, 1)
        t1 = th.sigmoid(th.randn(a.batch, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        t2 = th.sigmoid(th.randn(a.batch, device=dev))
        eps2 = th.randn_like(x0)
        x_t2 = (1 - t2[:, None, None, None]) * x0 + t2[:, None, None, None] * eps2
        v2_target = eps2 - x0

        with th.autocast("cuda", dtype=th.bfloat16):
            # 1. Stage 1: 单步速度预测 + Tweedie 投影
            v1 = gen(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple): v1 = v1[0]
            l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)

            # Tweedie 单步闭式流投影: g_pred = z_t - t * v1
            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)

            # 拉普拉斯拓扑连续性
            lap_pred = laplacian_filter(g_pred.float())
            lap_gt = laplacian_filter(g_gt.float())
            l_lap = th.nn.functional.mse_loss(lap_pred, lap_gt)

            # 2. Stage 2: 注入可微 g_pred, 反向回传图像真迹视觉梯度
            v2 = bak(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
            if isinstance(v2, tuple): v2 = v2[0]
            l_img = th.nn.functional.mse_loss(v2.float(), v2_target)

            # 复合联合损失
            loss = l_img + a.lam_skel_flow * l_sflow + a.lam_skel_mse * l_smse + a.lam_skel_lap * l_lap

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = th.nn.utils.clip_grad_norm_(gen_p, 1.0)
        opt.step()
        sched.step()
        step += 1

        # EMA 更新
        with th.no_grad():
            if step % a.ema_interval == 0:
                for kk, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[kk].copy_(v)

        r_img += float(l_img)
        r_sflow += float(l_sflow)
        r_smse += float(l_smse)
        r_lap += float(l_lap)
        r_gn += float(gn)
        cnt += 1

        if step % a.log_every == 0 or a.smoke:
            dt = time.time() - t0
            sps = cnt / max(dt, 1e-4)
            smp = sps * a.batch
            vram = th.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[step {step:05d}] L_img={r_img/cnt:.4f} L_flow={r_sflow/cnt:.4f} "
                  f"L_smse={r_smse/cnt:.4f} L_lap={r_lap/cnt:.4f} |gn|={r_gn/cnt:.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
            r_img = r_sflow = r_smse = r_lap = r_gn = 0.0
            cnt = 0
            t0 = time.time()

        if (step % a.ckpt_every == 0 or (a.smoke and step == max_steps)):
            _sd = _strip(gen.state_dict())
            _ema = _strip(ema)
            p_out = os.path.join(ckpt_dir, f"{step:07d}.pt")
            th.save({"gen": _sd, "gen_ema": _ema, "step": step, "args": vars(a)}, p_out)
            print(f"[ckpt] Checkpoint 已落盘: {p_out}", flush=True)

        if not a.smoke and step % a.eval_every == 0:
            run_eval(step)

    print(f"\n[done] 1-Step 联合微调全部完成 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
