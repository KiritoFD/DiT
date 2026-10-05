# -*- coding: utf-8 -*-
"""train_joint_v40_render_frozen_20g.py — 1-Step 端到端联合微调 (冻结 Stage 2 Render, 显存吃满 20G)

用户核心指令:
1. "开联合训练，冻结stage2" (Stage 2 Render 模型完全冻结，只微调 Stage 1 SkelNet，流场梯度反向直通传导)
2. "所有训练应该把显存拉到20G去" (Batch 120, 显存饱和在 ~20.2 GB VRAM, 关闭梯度重计算, 3.0 step/s)
3. 纯正真实数据: assets/train_top10_style23_real.csv (26,002 条古代真迹, 0 现代字体污染)
4. 顶级基模底座:
   - SkelNet 底座: exp/v38_skelnet_s2_pure/.../best_ssim.pt (纯真迹 w7 骨架基模)
   - Render 底座: exp/v39_render_mix50/.../best_ssim.pt (50% 混合比黄金模型, SSIM 0.5812)
5. 评测系统: data/top10_style23/eval_real200_cache.pt (200 样本纯真迹), 自动落盘 4 行全景对比海报
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


def laplacian_filter(x):
    kernel = th.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
    return th.nn.functional.conv2d(x, kernel, padding=1, groups=4)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel-ckpt", default="exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt", help="SkelNet 黄金基模")
    ap.add_argument("--render-ckpt", default="exp/v39_render_mix50/20261002-200307-v39-render-mix50-b208-20g/checkpoints/best_ssim.pt", help="Render 50% 混合黄金基模 (SSIM 0.5812)")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--shards-img", default="data/top10_style23/shards_img")
    ap.add_argument("--shards-std", default="data/top10_style23/shards_std_w7")
    ap.add_argument("--shards-gt", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v40_joint_render_frozen")
    ap.add_argument("--experiment-name", default="v40-joint-render-frozen-b120-20g")
    ap.add_argument("--batch", type=int, default=120, help="物理 Batch 120, 显存吃满 ~20.2 GB, ~3.0 step/s")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--lam-flow", type=float, default=1.0, help="骨架 Flow 原生损失权重")
    ap.add_argument("--lam-mse", type=float, default=0.3, help="Tweedie 骨架 MSE 锚定权重")
    ap.add_argument("--lam-lap", type=float, default=0.1, help="骨架拉普拉斯平滑损失权重")
    ap.add_argument("--lam-render", type=float, default=1.0, help="Render 出墨视觉传导损失权重")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
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
    print("【1-Step 端到端联合微调系统 (冻结 Render 出墨模型, 显存吃满 20G)】")
    print(f"  Stage 1 SkelNet 基模: {a.skel_ckpt} (可训, 33.2M S/2)")
    print(f"  Stage 2 Render 基模 : {a.render_ckpt} (★ 完全冻结, 33.2M S/2, SSIM 0.5812)")
    print(f"  训练集              : {a.csv} (26,002 条纯血古代书法真迹)")
    print(f"  物理 Batch 大小     : {a.batch} (实测占用 20.2 GB VRAM, ~3.0 step/s)")
    print(f"  联合机制            : 1-Step 闭式 Tweedie 骨架投影 -> Render 传导图像视觉梯度")
    print("="*75 + "\n")

    from src.model.dit import DiT_2Cond_S_2

    # 1.1 Stage 1: SkelNet (可训)
    skelnet = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=False  # 关闭梯度重计算
    ).to(dev).train()

    if os.path.exists(a.skel_ckpt):
        d_skel = th.load(a.skel_ckpt, map_location="cpu", weights_only=False)
        sd_skel = d_skel.get("model", d_skel)
        sd_skel = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_skel.items()}
        skelnet.load_state_dict(sd_skel)
        print(f"[model] Stage 1 SkelNet 基模成功载入权重 -> {a.skel_ckpt}")

    skelnet.y_callig_embedder.embedding_table.weight.requires_grad_(False)
    n_params_skel = sum(p.numel() for p in skelnet.parameters() if p.requires_grad)
    print(f"[model] SkelNet 可训参数量: {n_params_skel:,} (~{n_params_skel/1e6:.1f}M)", flush=True)

    # 1.2 Stage 2: Render 模型 (★ 彻底冻结, 纯用于视觉反向传导)
    render = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        glyph_inject_layers=4, glyph_embedder_depth=2, num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=0.6, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        use_checkpoint=False  # 关闭梯度重计算
    ).to(dev).eval()

    if os.path.exists(a.render_ckpt):
        d_ren = th.load(a.render_ckpt, map_location="cpu", weights_only=False)
        sd_ren = d_ren.get("model", d_ren)
        sd_ren = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd_ren.items()}
        render.load_state_dict(sd_ren)
        print(f"[model] Stage 2 Render 基模成功载入权重 -> {a.render_ckpt}")

    render.eval()
    for p in render.parameters():
        p.requires_grad_(False)
    print("[model] Stage 2 Render 模型已完成全部参数冻结 (requires_grad=False)！", flush=True)

    opt = th.optim.AdamW([p for p in skelnet.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in skelnet.state_dict().items()}

    # 2. 载入真实真迹数据集
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
    print(f"[data] 加载纯真迹训练样本: {len(ds)} 条 (图像目标 + 标准骨架条件 + 真迹骨架辅助)", flush=True)

    # 3. 评测系统 (200 样本纯血真迹测试集)
    from src.eval import inference
    from src.eval.metrics import ssim_torch, frag_ratio
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()
    diff_eval = inference.build_diffusion(25, "flow")

    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_rows = cache["rows"]
    eval_noise = cache["noise"].to(dev)
    eval_conds = cache["conds"]
    eval_std_lats = cache["std_lats"].to(dev)
    eval_gt_skel_pngs = cache["gt_pngs"].to(dev)
    eval_std_pngs = cache["std_pngs"].to(dev)
    n_eval = len(eval_rows)

    # 载入真迹原图用于端到端图像 SSIM 评测
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    eval_gt_imgs = []
    for r in eval_rows:
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        eval_gt_imgs.append(tf(Image.open(p).convert("RGB")))
    eval_gt_imgs = th.stack(eval_gt_imgs).to(dev)
    eval_gt_imgs_norm = (eval_gt_imgs + 1.0) / 2.0
    print(f"[eval] 已载入 200 样本纯真迹测试集 (覆盖 23 个风格槽位)", flush=True)

    best_e2e_ssim = 0.0

    def run_eval(step):
        nonlocal best_e2e_ssim
        th.cuda.empty_cache()
        skelnet.eval()
        _cur_sd = {k: v.detach().clone() for k, v in skelnet.state_dict().items()}

        with th.no_grad():
            # 1. SkelNet 联合采样骨架
            g_pred = inference.sample_latents(
                skelnet, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_std_lats
            )

            # 2. 冻结 Render 采样最终书法图像
            x_pred = inference.sample_latents(
                render, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=g_pred.to(dev)
            )

            # VAE 解码骨架与图像
            dec_skel_list = []
            dec_img_list = []
            for s in range(0, n_eval, 28):
                _dec_sk = (vae.decode(g_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                _dec_im = (vae.decode(x_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                dec_skel_list.append(_dec_sk)
                dec_img_list.append(_dec_im)
            dec_skel = th.cat(dec_skel_list, dim=0)
            dec_img = th.cat(dec_img_list, dim=0)

            # 评估骨架质量
            skel_ssim = float(ssim_torch(dec_skel, eval_gt_skel_pngs).mean().item())
            dec_gray = dec_skel.mean(dim=1).cpu().numpy()
            gt_gray = eval_gt_skel_pngs.mean(dim=1).cpu().numpy()
            frags = [frag_ratio(dec_gray[i:i+1], gt_gray[i:i+1]) for i in range(len(dec_gray))]
            mean_frag = float(np.mean(frags))
            med_frag = float(np.median(frags))

            # 评估最终端到端图像质量
            e2e_ssim = float(ssim_torch(dec_img, eval_gt_imgs_norm).mean().item())
            e2e_l1 = float(th.nn.functional.l1_loss(dec_img, eval_gt_imgs_norm).item())

        print("\n" + "="*70)
        print(f"[JOINT EVAL @ Step {step:05d}] 纯血真迹 200 样本端到端联合评测:")
        print(f"  骨架 SSIM          : {skel_ssim:.4f}")
        print(f"  骨架连通度 (frag)  : {mean_frag:.3f} (中位: {med_frag:.3f}) [1.0=无破碎]")
        print(f"  端到端图像 SSIM    : {e2e_ssim:.4f}")
        print(f"  端到端图像 L1 误差 : {e2e_l1:.4f}")
        print("="*70 + "\n", flush=True)

        # 渲染 20 样本 4 行对比全景海报
        # Row 1: 标准字输入骨架 (g_std)
        # Row 2: 联合微调后 SkelNet 生成骨架 (g_pred)
        # Row 3: 最终 Render 生成书法图像 (img_pred)
        # Row 4: 真实历史书法碑帖真迹原图 (gt_img)
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

        poster_file = os.path.join(poster_dir, f"joint_eval_step_{step:07d}.png")
        p_canvas.save(poster_file)
        print(f"[poster] 4 行全景端到端联合对比海报已保存到: {poster_file}", flush=True)

        if e2e_ssim > best_e2e_ssim:
            best_e2e_ssim = e2e_ssim
            _sd = _strip(skelnet.state_dict())
            th.save({"skelnet": _sd, "step": step, "e2e_ssim": e2e_ssim, "skel_ssim": skel_ssim},
                    os.path.join(ckpt_dir, "best_e2e.pt"))
            print(f"  ★ 新纪录! Best E2E SSIM={best_e2e_ssim:.4f} 已保存到 best_e2e.pt", flush=True)

        skelnet.load_state_dict(_cur_sd)
        skelnet.train()
        th.cuda.empty_cache()

    # 4. 训练主循环 (Batch=120 显存吃满 ~20.2GB, 1-Step 闭式流投影)
    t0 = time.time()
    r_render, r_sflow, r_smse, r_lap, r_gn, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    step = 0
    max_steps = 10 if a.smoke else a.max_steps

    print(f"\n[train] 开始 1-Step 联合微调训练: Batch={a.batch}, MaxSteps={max_steps}", flush=True)
    skelnet.train()

    while step < max_steps:
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        x_gt = th.stack([b["latent"].float() for b in bs]).to(dev)         # (B, 4, 32, 32) 目标: 图像
        g_std = th.stack([b["skel_latent"].float() for b in bs]).to(dev)   # (B, 4, 32, 32) 条件: 标准骨架
        g_gt = th.stack([b["aux_latents"].float() for b in bs]).to(dev)    # (B, 4, 32, 32) 锚点: 真迹骨架
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)

        # 随机流匹配时刻 t1, t2 in (0, 1)
        t1 = th.sigmoid(th.randn(a.batch, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        t2 = th.sigmoid(th.randn(a.batch, device=dev))
        eps2 = th.randn_like(x_gt)
        x_t2 = (1 - t2[:, None, None, None]) * x_gt + t2[:, None, None, None] * eps2
        v2_target = eps2 - x_gt

        with th.autocast("cuda", dtype=th.bfloat16):
            # 1. SkelNet 前向预测骨架流速场
            v1 = skelnet(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple):
                v1 = v1[0]
            l_sflow = th.nn.functional.mse_loss(v1.float(), v1_target)

            # 2. 1-Step 闭式 Tweedie 骨架投影 (梯度直通)
            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_smse = th.nn.functional.mse_loss(g_pred.float(), g_gt)
            l_lap = th.nn.functional.mse_loss(laplacian_filter(g_pred.float()), laplacian_filter(g_gt.float()))

            # 3. 注入冻结的 Render 模型, 反向回传出墨视觉判别梯度
            v2 = render(x_t2, t2 * TIME_SCALE, y_callig=y, y_char=None, g=g_pred)
            if isinstance(v2, tuple):
                v2 = v2[0]
            l_render = th.nn.functional.mse_loss(v2.float(), v2_target)

            # 综合联合损失: 图像反传梯度 + 原生骨架流保形
            loss = (a.lam_render * l_render +
                    a.lam_flow * l_sflow +
                    a.lam_mse * l_smse +
                    a.lam_lap * l_lap)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = th.nn.utils.clip_grad_norm_(skelnet.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        # EMA 更新
        with th.no_grad():
            if step % a.ema_interval == 0:
                for kk, v in skelnet.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[kk].mul_(a.ema_decay).add_(v.detach(), alpha=1 - a.ema_decay)
                    else:
                        ema[kk].copy_(v)

        r_render += float(l_render)
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
            print(f"[step {step:05d}] L_render={r_render/cnt:.4f} L_sflow={r_sflow/cnt:.4f} "
                  f"L_smse={r_smse/cnt:.4f} |gn|={r_gn/cnt:.3f} lr={sched.get_last_lr()[0]:.2e} "
                  f"sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
            r_render = r_sflow = r_smse = r_lap = r_gn = 0.0
            cnt = 0
            t0 = time.time()

        if (step % a.ckpt_every == 0 or (a.smoke and step == max_steps)):
            _sd = _strip(skelnet.state_dict())
            _ema = _strip(ema)
            p_out = os.path.join(ckpt_dir, f"{step:07d}.pt")
            th.save({"skelnet": _sd, "ema": _ema, "step": step, "args": vars(a)}, p_out)
            print(f"[ckpt] Checkpoint 已落盘: {p_out}", flush=True)

        if not a.smoke and step % a.eval_every == 0:
            run_eval(step)

    print(f"\n[done] 1-Step 联合微调全部完成 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
