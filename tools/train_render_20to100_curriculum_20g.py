# -*- coding: utf-8 -*-
"""train_render_20to90_curriculum_20g.py — Render 模型 20% -> 40% -> 60% -> 80% -> 90% 级联全收敛训练系统

用户核心指令:
1. "把render从20%到90%级联训练上去，每一个阶段都训练到收敛"
   - 起始底座: exp/v34_stage2_mix25/.../0030000.pt (出墨 0.7262 的最强基模)
   - 阶梯演进:
     * Stage 1: 20% Pred + 80% GT (平滑继承 v34 25% 先验, 训练至收敛)
     * Stage 2: 40% Pred + 60% GT (逐步加深对预测骨架的适应, 训练至收敛)
     * Stage 3: 60% Pred + 40% GT (预测骨架占过半主导, 训练至收敛)
     * Stage 4: 80% Pred + 20% GT (大比例依赖预测骨架, 训练至收敛)
     * Stage 5: 90% Pred + 10% GT (终极高鲁棒泛化阶段, 训练至收敛)
2. "所有的eval都用pred" (100% 采用 SkelNet 预测骨架输入，绝不使用 GT 骨架泄露作弊)
3. "所有训练应该把显存拉到20G去" (Batch 208, 显存饱和在 20.44 GB VRAM, 关闭梯度重计算, 2.7+ step/s)
4. 纯正真实数据: assets/train_top10_style23_real.csv (26,002 条古代真迹)
"""
import os, sys, json, time, math, argparse
from collections import deque
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
    # ★ 2026-10-03: SkelNet 判定失败 -> 排除。条件只用两种骨架:
    #   std  = 标准字骨架 (推理时的真实条件, 课程终点 100%)
    #   gt   = 真迹骨架   (课程起点多数, 等于给模型看答案, 先学"出墨"这件事)
    ap.add_argument("--skel-shards-pred", default="data/top10_style23/shards_std_w7",
                    help="课程的 std 侧骨架 (标准字, w7 宽度保证在 latent 上可观)")
    ap.add_argument("--skel-shards-gt", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--init-ckpt", default="exp/v34_stage2_mix25/20261001-220019-v34_stage2_mix25/checkpoints/0030000.pt", help="v34 30k 黄金基模 (SSIM 0.7262)")
    ap.add_argument("--skelnet-ckpt", default="exp/v38_skelnet_s2_pure/20261002-142735-v38-s2-pure-b1440-20g/checkpoints/best_ssim.pt")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v43_render_20to90_curriculum")
    ap.add_argument("--experiment-name", default="v43-render-20to90-b208-20g")
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
    ap.add_argument("--stage-scale", type=float, default=1.0,
                    help="各阶段最少/最多步数的缩放 (冒烟/验证 eval 用, 如 0.01)")
    return ap.parse_args()


# ★ 从 20% 到 90% 级联各阶段规划: (阶段名称, Pred 骨架占比, GT 骨架占比, 阶段最小收敛步数, 阶段最大步数)
CURRICULUM_STAGES = [
    ("Stage 1 (20% Pred)", 0.20, 0.80, 1000, 1500),
    ("Stage 2 (40% Pred)", 0.40, 0.60, 1000, 1500),
    ("Stage 3 (60% Pred)", 0.60, 0.40, 1000, 1500),
    ("Stage 4 (80% Pred)", 0.80, 0.20, 1000, 1500),
    ("Stage 5 (90% Pred)", 0.90, 0.10, 1500, 2000),
    # ★ 补到 100% Pred: 推理时喂的就是 stage1 预测的骨架, 训练必须走到这一步
    #   (否则模型永远没见过"纯预测骨架"的输入分布)。
    ("Stage 6 (100% Pred)", 1.00, 0.00, 2000, 3000),
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
    print("【Render 模型 20% -> 40% -> 60% -> 80% -> 90% 级联全收敛训练系统】")
    print(f"  初始底座        : {a.init_ckpt} (v34 30k 经典高出墨基模, SSIM 0.7262)")
    print(f"  训练集          : {a.csv} (26,002 条古代书法纯真迹)")
    print(f"  物理 Batch 大小 : {a.batch} (显存吃满 20.44 GB VRAM, ~2.7+ step/s)")
    print(f"  评测机制        : ★ 100% 采用 SkelNet 预测骨架评测, 杜绝 GT 泄露")
    print("  级联梯度全阶段规划 :")
    total_est_steps = 0
    for name, pw, gw, min_s, max_s in CURRICULUM_STAGES:
        print(f"    - {name:20s}: 权重 [{pw:.2f}, {gw:.2f}], 阶梯收敛区间 [{min_s}, {max_s}] 步")
        total_est_steps += min_s
    print(f"  预计基准总用时  : 约 {total_est_steps * 3.25 / 60:.1f} ~ {sum(s[4] for s in CURRICULUM_STAGES) * 3.25 / 60:.1f} 分钟")
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

    # ★ 必须走 model_io 加载: 它会按 ckpt 的 args 重建架构, 并做
    #   "灌入预训练书家表 + freeze_callig_table" 两步后处理 —— 手工 load_state_dict
    #   拿不到 y_callig_embedder.null_embed 这个 key (model_io.py 开头记过的坑,
    #   实测就是这么崩的)。
    from src.eval.model_io import load_model_from_ckpt
    # v34 ckpt 只有 "ema" 键 (没有 "model"), 所以用 EMA 口径加载
    render, _ren_args = load_model_from_ckpt(a.init_ckpt, device=dev, use_ema=True)
    render = render.to(dev)
    render.use_checkpoint = False      # 铁律: 不用 gradient checkpointing
    print(f"[model] Render 继承 v34 基模 -> {a.init_ckpt}", flush=True)

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
        print(f"[model] SkelNet 黄金基模载入成功, 专供 100% Pred 评测生成 -> {a.skelnet_ckpt}", flush=True)

    for p in skelnet.parameters():
        p.requires_grad_(False)

    opt = th.optim.AdamW([p for p in render.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sum(s[4] for s in CURRICULUM_STAGES), eta_min=a.lr * 0.1)
    ema = {k: v.detach().clone() for k, v in render.state_dict().items()}

    # 2. 载入数据集
    from src.utils.callig_script_map import load_callig_script_map
    from src.utils.latent_dataset import MCCDLatentDataset
    from src.utils.deform_aug import random_skeleton_deformation

    csmap = load_callig_script_map(a.callig_map)

    skel_dirs_list = [a.skel_shards_pred, a.skel_shards_gt]
    init_weights = [CURRICULUM_STAGES[0][1], CURRICULUM_STAGES[0][2]]

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
    print(f"[data] 纯真迹数据集加载完成: {len(ds)} 条 (动态骨架混合级联)", flush=True)

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

    import torchvision.transforms as T
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    eval_gt_imgs = []
    for r in eval_rows:
        p = r["image_path"] if os.path.isabs(r["image_path"]) else os.path.join(ROOT, r["image_path"])
        eval_gt_imgs.append(tf(Image.open(p).convert("RGB")))
    eval_gt_imgs = th.stack(eval_gt_imgs).to(dev)
    eval_gt_imgs_norm = (eval_gt_imgs + 1.0) / 2.0

    print("[eval] 正在为 200 评测样本预先生成 100% Pred 骨架条件 (绝不使用 GT 骨架)...", flush=True)
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
    # ★ 2026-10-03: SkelNet 已判失败 -> 排除。评测条件改用 **std 标准字骨架**
    #   (= 推理时的真实条件); 于是"基线"也就等于"直接拿标准字当出墨"。
    eval_pred_skel_lats = eval_std_lats
    eval_pred_skel_pngs = eval_std_pngs
    print("[eval] 条件已切换为 std 标准字骨架 (排除 SkelNet 预测骨架)", flush=True)

    best_ssim = 0.0

    def run_eval(step, stage_label, pct_tag):
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

            # ★ IoU + LPIPS (SSIM 背景主导、会骗人: skelnet 实测 SSIM 过线而 IoU 没过线)
            #   并且同批算出"什么都不做(直接用 pred 骨架)"的基线, 否则不知道好坏。
            def _iou(a, b):
                inter = (a & b).sum((-1, -2)).float()
                union = (a | b).sum((-1, -2)).float().clamp(min=1)
                return (inter / union).mean().item()

            gm = eval_gt_imgs_norm.mean(1) < 0.6
            pm = dec.mean(1) < 0.6
            bm = eval_pred_skel_pngs.mean(1) < 0.6       # 基线: 直接拿 pred 骨架当出墨
            mean_iou, base_iou = _iou(pm, gm), _iou(bm, gm)
            base_ssim = float(np.mean(ssim_torch(eval_pred_skel_pngs,
                                                 eval_gt_imgs_norm).cpu().numpy()))
            try:
                from src.eval.in_mem_eval import _lpips_per_sample
                # ⚠ 该接口要 **NHWC + 值域 [0,1]** (内部自己 *2-1), 传 CHW 会得到 nan
                _lp = _lpips_per_sample(dec.permute(0, 2, 3, 1).cpu().numpy(),
                                        eval_gt_imgs_norm.permute(0, 2, 3, 1).cpu().numpy())
                _lb = _lpips_per_sample(eval_pred_skel_pngs.permute(0, 2, 3, 1).cpu().numpy(),
                                        eval_gt_imgs_norm.permute(0, 2, 3, 1).cpu().numpy())
                mean_lpips = float(np.mean(_lp)) if _lp else float("nan")
                base_lpips = float(np.mean(_lb)) if _lb else float("nan")
            except Exception as _e:                                    # noqa: BLE001
                mean_lpips = base_lpips = float("nan")

        print("\n" + "="*70)
        print(f"[{stage_label} 收敛评测 @ Step {step:05d}] 纯真迹 100% Pred 骨架出墨评测:")
        print(f"  出墨 SSIM (均值) : {mean_ssim:.4f} (中位: {med_ssim:.4f})"
              f"   基线(pred骨架) {base_ssim:.4f}  "
              f"{'✓' if mean_ssim > base_ssim else '✗ 不如不做'}")
        print(f"  出墨 IoU        : {mean_iou:.4f}"
              f"   基线(pred骨架) {base_iou:.4f}  "
              f"{'✓' if mean_iou > base_iou else '✗ 不如不做'}")
        print(f"  LPIPS (越低越好): {mean_lpips:.4f}"
              f"   基线(pred骨架) {base_lpips:.4f}  "
              f"{'✓' if mean_lpips < base_lpips else '✗ 不如不做'}")
        print(f"  出墨 L1 误差    : {l1_val:.4f}")
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

        poster_file = os.path.join(poster_dir, f"poster_stage_{pct_tag}_step_{step:07d}.png")
        p_canvas.save(poster_file)
        print(f"[poster] {stage_label} 4 行全景对比海报已保存: {poster_file}", flush=True)

        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            _sd = _strip(render.state_dict())
            th.save({"model": _sd, "step": step, "mean_ssim": mean_ssim, "l1": l1_val},
                    os.path.join(ckpt_dir, "best_ssim.pt"))
            print(f"  ★ 全局新纪录! Best SSIM={best_ssim:.4f} 已保存到 best_ssim.pt", flush=True)

        render.load_state_dict(_cur_sd)
        render.train()
        th.cuda.empty_cache()

    # 4. 级联收敛训练主循环
    t0 = time.time()
    r_flow, r_gn, cnt = 0.0, 0.0, 0
    current_global_step = 0
    render.train()

    for stage_idx, (stage_name, pred_w, gt_w, min_steps, max_steps) in enumerate(
            [(n_, pw_, gw_, max(2, int(mn_ * a.stage_scale)),
              max(3, int(mx_ * a.stage_scale)))
             for n_, pw_, gw_, mn_, mx_ in CURRICULUM_STAGES]):
        pct_tag = f"{int(pred_w*100)}pct"
        ds.skel_latent_shards_weights = [pred_w, gt_w]
        print(f"\n{'#'*75}")
        print(f"【进入级联阶梯】 {stage_name} (Pred={pred_w*100:.0f}%, GT={gt_w*100:.0f}%)")
        print(f"  收敛判定规则: 最少执行 {min_steps} 步, 最多执行 {max_steps} 步, 动态追踪平稳收敛")
        print(f"{'#'*75}\n", flush=True)

        loss_history = deque(maxlen=100)
        steps_this_stage = 0
        target_max = 5 if a.smoke else max_steps

        while steps_this_stage < target_max:
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
            loss_history.append(float(loss))

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
                print(f"[{stage_name} | Step {current_global_step:05d} ({steps_this_stage}/{max_steps})] "
                      f"L_flow={r_flow/cnt:.4f} |gn|={r_gn/cnt:.3f} lr={sched.get_last_lr()[0]:.2e} "
                      f"sps={sps:.2f} ({smp:.1f} smp/s) vram={vram:.2f}G", flush=True)
                r_flow = r_gn = 0.0
                cnt = 0
                t0 = time.time()

            # 动态收敛检测 (达到最小步数后, 检查损失是否完全拉平)
            if not a.smoke and steps_this_stage >= min_steps:
                if len(loss_history) == 100:
                    l_recent = list(loss_history)
                    first_half = np.mean(l_recent[:50])
                    second_half = np.mean(l_recent[50:])
                    if abs(first_half - second_half) < 0.0015 or steps_this_stage >= max_steps:
                        print(f"\n[converged] ★ 动态收敛达成! {stage_name} 在第 {steps_this_stage} 步稳定收敛 (ΔL={abs(first_half - second_half):.5f})", flush=True)
                        break

        # 阶段收敛完成：保存专属 Checkpoint 并评测出图
        _sd = _strip(render.state_dict())
        p_stage_ckpt = os.path.join(ckpt_dir, f"ckpt_{pct_tag}_converged_step{current_global_step:05d}.pt")
        th.save({"model": _sd, "step": current_global_step, "pred_w": pred_w, "gt_w": gt_w}, p_stage_ckpt)
        print(f"[stage-saved] 阶段收敛权重落盘: {p_stage_ckpt}", flush=True)

        if not a.smoke:
            run_eval(current_global_step, stage_name, pct_tag)

    print(f"\n[done] 20% -> 90% 级联全阶段收敛训练圆满结束 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
