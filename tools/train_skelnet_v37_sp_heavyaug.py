# -*- coding: utf-8 -*-
"""train_skelnet_v37_sp_heavyaug.py — SkelNet v3 纯血真迹强泛化基模训练:

  核心设计决策:
  1. 纯血真迹数据 (已彻底清洗删除全部 12,381 条劣质现代字库合成数据):
     - 训练集: assets/train_top10_style23_real.csv (26,002 条纯正古代碑帖墨迹真迹)
     - 黄金评测集: data/top10_style23/eval_real200_cache.pt (分层隔离的 200 条无偏置纯真迹)
  2. 模型容量: 用户指定的 DiT-2Cond-Sp/2 (hidden=512, depth=12, heads=8, ~65.4M 参数, 1.8x 表征带宽)
  3. 强力多维扰动 (彻底摧毁视觉哈希记忆捷径):
     - GPU 随机弹性形变与仿射扭曲 (prob=0.85, rot=8°, elastic=2.5px, 伴随保幅校准)
     - 局部方块抹白掩码 (prob=0.35, 强制模型学会笔画自愈与拓扑缝合)
     - 高斯潜变量抖动 (prob=0.40, scale=0.15)
  4. 复合监督信号 (吸收历史实验经验):
     - L_flow_whiten: 逐通道方差白化流匹配损失 (抑制背景通道支配)
     - L_dir: 全局余弦方向对齐损失 (权重 0.2, 防止幅度相消均值漂白)
     - L_mag: 向量模长相对误差损失 (权重 0.1, 治潜变量幅度收缩)
     - L_rec: Tweedie 干净骨架闭式重建损失 (权重 0.3, g_0 锚定)
     - L_lap: 空间二阶拉普拉斯拓扑连续性损失 (权重 0.1, 强烈惩罚笔画断裂端点)
     - loss_ramp: 前 1000 步平滑爬坡, 保证初始化平稳
  5. 7px 粗骨架目标与输入 (shards_gtskel_w7 + shards_std_w7)
"""
import os, sys, json, glob, csv, re, time, math, argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
sys.stdout.reconfigure(encoding="utf-8")
TIME_SCALE = 1000.0


def _strip(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std_w7")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt", help="200 条真迹预计算评测缓存")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v37_skelnet_sp")
    ap.add_argument("--experiment-name", default="v37-sp-real26k")
    ap.add_argument("--batch", type=int, default=128, help="DiT-Sp batch=128 仅占 4.5GB 显存，极速吞吐")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    # ── 监督信号权重 ──
    ap.add_argument("--lam-whiten", type=float, default=1.0)
    ap.add_argument("--lam-dir", type=float, default=0.2, help="余弦方向对齐权重")
    ap.add_argument("--lam-mag", type=float, default=0.1, help="向量模长相对误差权重")
    ap.add_argument("--lam-rec", type=float, default=0.3, help="Tweedie g_0 重建损失权重")
    ap.add_argument("--lam-lap", type=float, default=0.1, help="空间拉普拉斯拓扑连续性损失权重")
    ap.add_argument("--loss-ramp", type=int, default=1000, help="辅助损失平滑爬坡步数")
    # ── 强扰动增强超参 ──
    ap.add_argument("--deform-prob", type=float, default=0.85, help="几何弹性形变概率")
    ap.add_argument("--mask-prob", type=float, default=0.35, help="局部方块抹白概率")
    ap.add_argument("--noise-prob", type=float, default=0.40, help="高斯潜变量抖动概率")
    # ── 运行与评估 ──
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


# ── 拉普拉斯连续性算子 (针对 4 通道潜变量特征做空间二阶梯度) ──
def laplacian_filter(x):
    # x: (B, 4, H, W)
    kernel = th.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3).repeat(4, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=4)


# ── 骨架强扰动引擎 (集成几何变形 + 局部抹白 + 潜空间噪波) ──
def apply_heavy_perturbation(g, deform_prob, mask_prob, noise_prob, z_bg):
    # 1. 几何弹性形变增强 (使用 src/utils/deform_aug.py 原生算子)
    if deform_prob > 0:
        from src.utils.deform_aug import random_skeleton_deformation
        g = random_skeleton_deformation(
            g, prob=deform_prob, max_rot_deg=8.0, max_scale=0.15, max_shear=0.10,
            max_trans_px=2.0, max_elastic_px=2.5, coarse_res=6, preserve_amplitude=True
        )

    # 2. 局部方块掩码抹白 (Masked Skeleton Modeling: 强制愈合与缝合断线)
    B, C, H, W = g.shape
    device = g.device
    if mask_prob > 0:
        hit = (th.rand(B, device=device) < mask_prob)
        if hit.any():
            msk = th.zeros(B, 1, H, W, device=device, dtype=th.bool)
            for b in range(B):
                if not bool(hit[b]): continue
                # 随机挖掉 2~3 个 3x3~5x5 的小方块
                for _ in range(np.random.randint(2, 4)):
                    sz = np.random.randint(3, 6)
                    yy = np.random.randint(0, H - sz + 1)
                    xx = np.random.randint(0, W - sz + 1)
                    msk[b, 0, yy:yy+sz, xx:xx+sz] = True
            g = th.where(msk, z_bg.expand_as(g), g)

    # 3. 高斯潜变量抖动
    if noise_prob > 0:
        hit_noise = (th.rand(B, 1, 1, 1, device=device) < noise_prob)
        jitter = th.randn_like(g) * 0.15
        g = g + hit_noise.float() * jitter

    return g


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

    from src.utils.callig_script_map import load_callig_script_map
    csmap = load_callig_script_map(a.callig_map)

    # ── 数据集加载 ──
    from src.utils.latent_dataset import MCCDLatentDataset
    ds = MCCDLatentDataset(
        csv_file=a.csv, latent_shards_dir=a.tgt_shards, img_root="",
        image_size=256, is_train=True, preload=True, load_image=False,
        skel_latent_shards_dir=a.cond_shards,
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] 加载纯真迹训练样本: {len(ds)} 条 (目标: w7, 条件: w7)", flush=True)

    # ── 构建 DiT-2Cond-Sp/2 强基模 (~65.4M 参数) ──
    from src.model.dit import DiT_2Cond_Sp_2
    model = DiT_2Cond_Sp_2(
        callig_embed_dim=128,
        glyph_vec_cond=True,
        glyph_vec_dim=128,
        condition_fusion="factorized_cat",
        cond_fusion_norm="split",
        glyph_inject_layers=4,
        glyph_embedder_depth=2,
        num_calligraphers=23,
        use_glyph_cond=True,
        use_char_cond=False,
        learn_sigma=False,
        glyph_scale_init=0.6,
        norm_type="rms",
        mlp_type="swiglu",
        qk_norm=1,
        rope=1,
        rope_theta=100.0
    ).to(dev)

    if os.path.exists(a.style_emb):
        d = th.load(a.style_emb, map_location="cpu", weights_only=False)
        emb = d["embedding"] if isinstance(d, dict) else d
        with th.no_grad():
            w = model.y_callig_embedder.embedding_table.weight
            w[:emb.shape[0]].copy_(emb.float())
        print(f"[model] 已载入预训练风格表: {emb.shape}", flush=True)

    # 冻结风格表 [0, N)
    model.y_callig_embedder.embedding_table.weight.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DiT-2Cond-Sp/2 实例化完成: 可训参数量 {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    from src.utils.deform_aug import Z_BG_VEC
    z_bg = Z_BG_VEC.to(device=dev, dtype=th.bfloat16)

    # 估计各通道方差以做白化加权 (抑制白背景通道过度主导)
    _idx = np.random.choice(len(ds), min(512, len(ds)), replace=False)
    _A = th.stack([ds[int(i)]["latent"].float() for i in _idx])
    _sd = _A.reshape(_A.shape[0], _A.shape[1], -1).std(dim=(0, 2)).clamp_min(1e-4)
    whiten_w = (1.0 / _sd) / (1.0 / _sd).mean()
    whiten_weights = whiten_w.view(1, 4, 1, 1).to(dev)
    print(f"[loss] 逐通道白化权重: {[round(float(x), 3) for x in whiten_weights.flatten()]}", flush=True)

    # ── 评测系统 (直接读取预计算黄金 200 样本缓存) ──
    from src.eval import inference
    from src.eval.metrics import ssim_torch, frag_ratio
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()
    diff_eval = inference.build_diffusion(25, "flow")

    if not os.path.exists(a.eval_cache):
        raise FileNotFoundError(f"Eval cache not found: {a.eval_cache}. Run tools/build_eval_real200_cache.py first!")

    eval_data = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_noise = eval_data["noise"].to(dev)
    eval_conds = eval_data["conds"]
    eval_std_lats = eval_data["std_lats"].to(dev)
    eval_gt_pngs = eval_data["gt_pngs"].to(dev)
    eval_std_pngs = eval_data["std_pngs"].to(dev)
    n_eval = len(eval_conds)
    print(f"[eval] 成功载入预计算真迹评测集: {n_eval} 条样本", flush=True)

    best_ssim = 0.0
    best_frag = 999.0

    def run_eval(step):
        nonlocal best_ssim, best_frag
        model.eval()
        _cur_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in ema.items()})

        # 运行 25-step flow 采样 (batch=50, 4个批次瞬间完成)
        with th.no_grad():
            g_pred = inference.sample_latents(
                model, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_std_lats
            )
            # 分批 decode VAE (batch=28 防 VRAM 突刺)
            dec_list = []
            for s in range(0, n_eval, 28):
                _dec = (vae.decode(g_pred[s:s+28].to(dev) / 0.18215).sample.clamp(-1, 1) + 1) / 2
                dec_list.append(_dec)
            dec = th.cat(dec_list, dim=0)

            # 1. 骨架 SSIM (均值与中位数)
            ssim_vals = ssim_torch(dec, eval_gt_pngs).cpu().numpy()
            mean_ssim = float(np.mean(ssim_vals))
            med_ssim = float(np.median(ssim_vals))

            # 2. 墨迹 SSIM (骨架二值掩码一致性)
            dec_gray = dec.mean(dim=1)
            gt_gray = eval_gt_pngs.mean(dim=1)
            dec_mask = (dec_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
            gt_mask = (gt_gray < 0.6).float().unsqueeze(1).repeat(1, 3, 1, 1)
            ink_ssim_vals = ssim_torch(dec_mask, gt_mask).cpu().numpy()
            mean_ink = float(np.mean(ink_ssim_vals))

            # 3. 骨架连通块破碎度 (连通块数量比，越接近 1.0 越连贯)
            p_np = dec_gray.cpu().numpy()
            g_np = gt_gray.cpu().numpy()
            frags = [frag_ratio(p_np[i:i+1], g_np[i:i+1]) for i in range(len(p_np))]
            mean_frag = float(np.mean(frags))
            med_frag = float(np.median(frags))

        print(f"\n" + "="*65, flush=True)
        print(f"[EVAL @ Step {step:05d}] 纯血真迹 200 样本评测:", flush=True)
        print(f"  骨架 SSIM (均值)   : {mean_ssim:.4f} (中位: {med_ssim:.4f})", flush=True)
        print(f"  墨迹 SSIM (Ink)    : {mean_ink:.4f}", flush=True)
        print(f"  破碎度 (frag_ratio): {mean_frag:.3f} (中位: {med_frag:.3f}, 越接近 1.0 越无断裂!)", flush=True)
        print("="*65 + "\n", flush=True)

        # 4. 渲染可视化海报 (抽样 20 个样本横跨不同风格: 标准输入 vs 生成骨架 vs 真实真迹)
        p_cols = 20
        p_canvas = Image.new("RGB", (256 * p_cols, 256 * 3))
        # 均匀抽 20 个样本
        sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
        for col_idx, col in enumerate(sub_indices):
            p_std = Image.fromarray((eval_std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gen = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gt = Image.fromarray((eval_gt_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_canvas.paste(p_std, (col_idx * 256, 0))
            p_canvas.paste(p_gen, (col_idx * 256, 256))
            p_canvas.paste(p_gt, (col_idx * 256, 512))
        p_poster = os.path.join(poster_dir, f"eval_step_{step:07d}.png")
        p_canvas.save(p_poster)

        # 记录最优模型
        _sd = _strip(model.state_dict())
        _ema = _strip(ema)
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            th.save({"model": _sd, "ema": _ema, "gen": _sd, "gen_ema": _ema,
                     "step": step, "args": vars(a), "mean_ssim": mean_ssim, "mean_frag": mean_frag},
                    os.path.join(ckpt_dir, "best_ssim.pt"))
            print(f"  ★ 新纪录! Best SSIM={best_ssim:.4f} 已保存到 best_ssim.pt", flush=True)

        if mean_frag < best_frag:
            best_frag = mean_frag
            th.save({"model": _sd, "ema": _ema, "gen": _sd, "gen_ema": _ema,
                     "step": step, "args": vars(a), "mean_ssim": mean_ssim, "mean_frag": mean_frag},
                    os.path.join(ckpt_dir, "best_frag.pt"))
            print(f"  ★ 新纪录! Best Frag={best_frag:.3f} 已保存到 best_frag.pt", flush=True)

        model.load_state_dict(_cur_sd)
        model.train()
        return mean_ssim, mean_frag

    # ── 主训练循环 ──
    t0 = time.time()
    r_flow, r_dir, r_mag, r_rec, r_lap, r_gn, cnt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    step = 0
    max_steps = 10 if a.smoke else a.max_steps

    print(f"\n[train] 开始训练: Batch={a.batch}, MaxSteps={max_steps}", flush=True)
    model.train()

    while step < max_steps:
        k = np.random.randint(0, len(ds), a.batch)
        bs = [ds[int(j)] for j in k]
        g_gt = th.stack([b["latent"].float() for b in bs]).to(dev)       # (B, 4, 32, 32)
        g_std = th.stack([b["skel_latent"].float() for b in bs]).to(dev)  # (B, 4, 32, 32)
        y = th.tensor([int(b["y_callig"]) for b in bs], dtype=th.long, device=dev)

        # ★ 施加强力几何与拓扑扰动 (破除特征哈希死记硬背)
        g_std_aug = apply_heavy_perturbation(g_std, a.deform_prob, a.mask_prob, a.noise_prob, z_bg)

        # 随机流匹配时刻 t1 in (0, 1)
        t1 = th.sigmoid(th.randn(a.batch, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        # 平滑爬坡因子
        ramp = min(1.0, float(step) / max(float(a.loss_ramp), 1.0))

        with th.autocast("cuda", dtype=th.bfloat16):
            # 1. 速度场预测 (use_char_cond=False 时 y_char 传 None 即可)
            v1 = model(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std_aug)
            if isinstance(v1, tuple):
                v1 = v1[0]

            # 2. 逐通道白化流匹配损失
            diff_sq = (v1.float() - v1_target.float()) ** 2
            l_flow = (diff_sq * whiten_weights).mean()

            # 3. 余弦方向对齐与模长保护 (压平向量计算，防止局部 0 梯度震荡)
            fb = v1.reshape(v1.shape[0], -1).float()
            tb = v1_target.reshape(v1_target.shape[0], -1).float()
            nf = fb.norm(dim=1).clamp_min(1e-4)
            nt = tb.norm(dim=1).clamp_min(1e-4)
            cos_sim = (fb * tb).sum(dim=1) / (nf * nt)
            l_dir = (1.0 - cos_sim).mean()
            l_mag = ((nf - nt).pow(2) / nt.pow(2)).mean()

            # 4. Tweedie g_0 端点闭式重建
            g_pred = z_t1 - t1[:, None, None, None] * v1
            l_rec = F.mse_loss(g_pred.float(), g_gt)

            # 5. 空间拉普拉斯拓扑连续性损失 (严惩笔画断裂端点)
            lap_pred = laplacian_filter(g_pred.float())
            lap_gt = laplacian_filter(g_gt.float())
            l_lap = F.mse_loss(lap_pred, lap_gt)

            # 复合总损失 (平滑爬坡)
            loss = (a.lam_whiten * l_flow +
                    ramp * (a.lam_dir * l_dir +
                            a.lam_mag * l_mag +
                            a.lam_rec * l_rec +
                            a.lam_lap * l_lap))

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

        r_flow += float(l_flow)
        r_dir += float(l_dir)
        r_mag += float(l_mag)
        r_rec += float(l_rec)
        r_lap += float(l_lap)
        r_gn += float(gn)
        cnt += 1

        if step % a.log_every == 0 or a.smoke:
            dt = time.time() - t0
            sps = cnt / max(dt, 1e-4)
            smp = sps * a.batch
            vram = th.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[step {step:05d}] L_flow={r_flow/cnt:.4f} "
                  f"L_dir={r_dir/cnt:.4f} L_mag={r_mag/cnt:.4f} L_rec={r_rec/cnt:.4f} L_lap={r_lap/cnt:.4f} "
                  f"|gn|={r_gn/cnt:.3f} lr={sched.get_last_lr()[0]:.2e} "
                  f"sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
            r_flow = r_dir = r_mag = r_rec = r_lap = r_gn = 0.0
            cnt = 0
            t0 = time.time()

        if (step % a.ckpt_every == 0 or (a.smoke and step == max_steps)):
            _sd = _strip(model.state_dict())
            _ema = _strip(ema)
            p_out = os.path.join(ckpt_dir, f"{step:07d}.pt")
            th.save({"model": _sd, "ema": _ema,
                     "gen": _sd, "gen_ema": _ema,
                     "step": step, "args": vars(a)}, p_out)
            print(f"[ckpt] Checkpoint 已落盘: {p_out}", flush=True)

        if step % a.eval_every == 0 or (a.smoke and step == max_steps):
            run_eval(step)

    # 训练结束最终评测
    if not a.smoke and step % a.eval_every != 0:
        print("\n[done] 运行训练结束最终全面评测...", flush=True)
        run_eval(step)

    print(f"\n[done] SkelNet-Sp 纯血真迹强泛化训练全部完成 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
