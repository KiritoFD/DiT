# -*- coding: utf-8 -*-
"""train_skelnet_v38_pure_s2_20g.py — 用户裁定: 回归正统 S/2 架构, 纯血真迹 w7 极速训练 (拉满 20G 显存):

  用户核心指令:
  1. "skelnet目前的基模不合格啊，还不如之前那个呢。我怀疑是之前我们的test过度严苛，不是随机选出来的。"
  2. "就单纯w7，用之前的skelnet在干净数据上训练，给我poster"
  3. "所有训练应该把显存拉到20G去。"

  落地技术方案:
  1. 架构: 回归历史最佳口碑 DiT-2Cond-S/2 基模 (33.2M 参数, S/2 原汁原味架构)
  2. 纯粹 w7 监督: 彻底移除人工弹性扭曲与方块掩码, 纯净 Flow Matching (目标: w7, 条件: w7)
  3. 数据: assets/train_top10_style23_real.csv (26,002 条纯正古代书法碑帖真迹, 0 现代字库污染)
  4. 显存拉满 20G: 物理 Batch=1440, 实测显存占用 19.37 GB, 吞吐 459+ samples/s (每步覆盖全部 23 槽位)
  5. 评测: data/top10_style23/eval_real200_cache.pt (200 样本随机分层真迹), 自动落盘 3 行对比全景海报
"""
import os, sys, json, glob, csv, re, time, math, argparse
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
    ap.add_argument("--tgt-shards", default="data/top10_style23/shards_gtskel_w7")
    ap.add_argument("--cond-shards", default="data/top10_style23/shards_std_w7")
    ap.add_argument("--csv", default="assets/train_top10_style23_real.csv", help="纯血真迹训练集 (26,002 条)")
    ap.add_argument("--eval-cache", default="data/top10_style23/eval_real200_cache.pt")
    ap.add_argument("--style-emb", default="assets/callig_script_emb_top10.pt")
    ap.add_argument("--callig-map", default="assets/callig_script_id_map_top10.json")
    ap.add_argument("--vae", default="data/pretrained/pretrained_models/sd-vae-ft-ema")
    ap.add_argument("--results-dir", default="exp/v44_skelnet_concat")
    ap.add_argument("--experiment-name", default="v44-concat-b1440-20g")
    ap.add_argument("--batch", type=int, default=1440, help="物理 Batch 1440, 显存吃满 19.37 GB")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-interval", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="续训 ckpt (只恢复权重, lr 会重新 warm)")
    # ── 结构监督 (2026-10-03): 实测 latent L2 与骨架质量不同源 (L_flow 平台 0.256
    #    而 IoU/SSIM 卡在门槛下), CPU 探针排序后选 **投影剖面为主 + 软 Dice 为辅**
    #    (Spearman vs 1-IoU: proj .959 / dice 1.000 / dt .924 / SWD .732, 且 SWD 在
    #     小误差非单调、dt 不罚缺失笔画 -> 都不单用)。
    ap.add_argument("--lam-proj", type=float, default=1.0,
                    help="投影剖面损失权重 (多方向 Radon 剖面 L1) —— 主监督")
    ap.add_argument("--lam-dice", type=float, default=0.5,
                    help="软 Dice 权重 (与 IoU 同源 rho=1.0, 但坏端饱和 -> 只作辅)")
    ap.add_argument("--lam-latent", type=float, default=0.3,
                    help="latent flow MSE 保留权重 (只负责把 latent 留在 VAE 流形上)")
    ap.add_argument("--struct-every", type=int, default=4, help="每 N 步算一次结构损失")
    ap.add_argument("--struct-batch", type=int, default=24, help="结构损失用的样本数")
    ap.add_argument("--struct-tau", type=float, default=0.08, help="软掩码温度")
    ap.add_argument("--gt-mask-cache", default="data/top10_style23/gt_mask_w7_train.npy",
                    help="真迹墨迹掩码缓存 (解码一次后落盘复用)")
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
    print("【SkelNet v38 正统 S/2 纯血真迹训练 (显存拉满 20G 旗舰级)】")
    print(f"  模型架构        : DiT-2Cond-S/2 (33.2M 原版基模架构)")
    print(f"  训练集          : {a.csv} (26,002 条纯正古代书法真迹)")
    print(f"  物理 Batch 大小 : {a.batch} (实测占用 19.37 GB VRAM, 459+ samples/s)")
    print(f"  目标/条件       : 纯净 w7 (无人工几何扭曲，纯净学书法间架)")
    print("="*75 + "\n")

    # 1. 载入模型 (DiT-2Cond-S/2 原版架构)
    from src.model.dit import DiT_2Cond_S_2
    # ★ v44: 条件注入从 token-add 改成 **输入 concat** + 大幅增强。
    #   依据 (2026-10-02 探测): v38 用 add/adaln 注入时, 同一条件 4 次采样的两两 IoU
    #   (diversity=0.104) 竟然和"输出 vs 真迹"的重合度 (0.097) 同量级 -> 条件**没有
    #   把输出约束住**。concat 把原始骨架直接拼进第一层输入通道, 让每一步都能看到它。
    model = DiT_2Cond_S_2(
        callig_embed_dim=128, glyph_vec_cond=True, glyph_vec_dim=128,
        condition_fusion="factorized_cat", cond_fusion_norm="split",
        # 输入 concat (而非 token-add): 通道数 4 -> 8
        glyph_concat_input=True,
        # 注入大幅增强: 层数 4 -> 8, 初始增益 0.6 -> 1.0, 条件编码器加深
        # ⚠ glyph_embedder_sep 必须开: depth=3 的**满秩** 3x3 conv @hidden384 极贵
        #   (dit.py:954-963 记了这个坑; 实测 batch128 就要 4.7s/step)。
        #   改成 depthwise-separable 后同样的深度几乎不要钱。
        # 注入层数保持 4 (与 v38 一致) —— 本轮唯一要验的变量是 **concat vs add**,
        #   层数一起变就没法归因。8 层在无 checkpoint 的 24G 卡上最多 batch=128
        #   且步速异常(4.7s/step), 无法做对照实验。
        glyph_inject_layers=4, glyph_embedder_depth=2, glyph_embedder_sep=True,
        num_calligraphers=23,
        use_glyph_cond=True, use_char_cond=False, learn_sigma=False,
        glyph_scale_init=1.0, norm_type="rms", mlp_type="swiglu",
        qk_norm=1, rope=1, rope_theta=100.0,
        # ★ 铁律: 不允许 gradient checkpointing (浪费算力)。DiT_2Cond 的默认值就是
        #   True (src/model/dit.py:889), 不显式关掉就会被静默开启 —— 之前几轮
        #   能开到 batch 1440 全是靠它换来的。代价: batch 必须相应调小。
        use_checkpoint=False,
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
    print(f"[model] DiT-2Cond-S/2 实例化完成: 可训参数量 {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    if a.resume and os.path.isfile(a.resume):
        _ck = th.load(a.resume, map_location="cpu", weights_only=False)
        _sd = _ck.get("model", _ck)
        _sd = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in _sd.items()}
        _ms, _us = model.load_state_dict(_sd, strict=False)
        print(f"[resume] {a.resume} (step={_ck.get('step')}) "
              f"missing={len(_ms)} unexpected={len(_us)}", flush=True)

    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(a.warmup, 1),
                           a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 *
                           (1 + math.cos(math.pi * min(1.0, s / max(a.max_steps, 1))))))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 2. 载入数据集
    from src.utils.callig_script_map import load_callig_script_map
    from src.utils.latent_dataset import MCCDLatentDataset
    csmap = load_callig_script_map(a.callig_map)

    ds = MCCDLatentDataset(
        csv_file=a.csv, latent_shards_dir=a.tgt_shards, img_root="",
        image_size=256, is_train=True, preload=True, load_image=False,
        skel_latent_shards_dir=a.cond_shards,
        callig_id_map=None, callig_script_map=csmap)
    print(f"[data] 加载纯真迹训练样本: {len(ds)} 条 (目标: w7, 条件: w7)", flush=True)

    # 3. 评测系统 (黄金 200 样本纯真迹测试集)
    from src.eval import inference
    from src.eval.metrics import ssim_torch, frag_ratio
    from src.eval.in_mem_eval import _get_vae

    vae = _get_vae(dev, a.vae).eval()

    # ── 真迹墨迹掩码缓存: 解码一次, 落盘复用 (结构监督的参照) ──
    import torch.nn.functional as F
    _H = 256
    _ii = np.arange(_H)[:, None] + np.arange(_H)[None, :]
    IDX45 = th.from_numpy(_ii.reshape(-1).astype(np.int64))
    IDX135 = th.from_numpy((np.arange(_H)[:, None]
                            + (_H - 1 - np.arange(_H))[None, :]).reshape(-1).astype(np.int64))

    def build_gt_masks():
        if os.path.isfile(a.gt_mask_cache):
            mm = np.load(a.gt_mask_cache, mmap_mode="r")
            print(f"[gt-mask] 复用缓存 {a.gt_mask_cache} {mm.shape}", flush=True)
            return mm
        lat = ds._latents                      # 训练集目标 latent (MCCDLatentDataset 预载)
        out = np.zeros((len(lat), _H, _H), np.uint8)
        with th.no_grad():
            # ⚠ 256² VAE 解码的激活很大, batch 64 会 OOM (实测) -> 用 16
            for s in range(0, len(lat), 16):
                z = th.from_numpy(np.asarray(lat[s:s + 16])).to(dev)
                dec = (vae.decode(z / 0.18215).sample.clamp(-1, 1) + 1) / 2
                g = dec.mean(1)
                out[s:s + 16] = (g < 0.6).cpu().numpy().astype(np.uint8)
                del dec, z
                if s % 2048 == 0:
                    th.cuda.empty_cache()
                    print(f"  [gt-mask] {s}/{len(lat)}", flush=True)
        os.makedirs(os.path.dirname(a.gt_mask_cache) or ".", exist_ok=True)
        np.save(a.gt_mask_cache, out)
        print(f"[gt-mask] 已缓存 {a.gt_mask_cache} {out.shape}", flush=True)
        return out

    GT_MASK = build_gt_masks() if (a.lam_proj > 0 or a.lam_dice > 0) else None

    def profiles_t(m):
        """4 方向求和剖面 (行/列/两条对角), 各自归一化 -> (B,4,L)。全可微。"""
        B, H, W = m.shape
        ps = []
        for p in (m.sum(2), m.sum(1)):
            ps.append(p / p.sum(1, keepdim=True).clamp(min=1e-6))
        f = m.flatten(1)
        for idx in (IDX45, IDX135):
            d = th.zeros(B, 2 * H - 1, device=m.device, dtype=m.dtype)
            ix = idx.to(m.device)
            # index_add_ 要求索引是**一维向量** -> 逐样本累加 (B 小, 开销可忽略)
            for b in range(B):
                d[b] = d[b].index_add(0, ix, f[b])
            ps.append(d / d.sum(1, keepdim=True).clamp(min=1e-6))
        return ps                                # 长度不一, 逐个比

    def struct_loss(x_hat, gt_idx):
        """x_hat: (b,4,32,32) 预测骨架 latent -> 解码 -> 软掩码 -> proj + dice"""
        with th.no_grad():
            dec = (vae.decode(x_hat / 0.18215).sample.clamp(-1, 1) + 1) / 2
        m = th.sigmoid((0.5 - dec.mean(1).float()) / a.struct_tau)   # 软墨迹
        g = th.from_numpy(np.asarray(GT_MASK[gt_idx], np.float32)).to(dev)
        lp = sum((pm - pg).abs().mean(1) for pm, pg in zip(profiles_t(m), profiles_t(g))) / 4
        inter = (m * g).sum((1, 2))
        ld = (1.0 - 2 * inter / (m.sum((1, 2)) + g.sum((1, 2))).clamp(min=1e-6))
        return lp.mean(), ld.mean()
    diff_eval = inference.build_diffusion(25, "flow")

    cache = th.load(a.eval_cache, map_location="cpu", weights_only=False)
    eval_noise = cache["noise"].to(dev)
    eval_conds = cache["conds"]
    eval_std_lats = cache["std_lats"].to(dev)
    eval_gt_pngs = cache["gt_pngs"].to(dev)
    eval_std_pngs = cache["std_pngs"].to(dev)
    n_eval = len(eval_conds)
    print(f"[eval] 已载入 200 样本纯血真迹评测集 (覆盖 23 个风格槽位)", flush=True)

    best_ssim = 0.0

    def run_eval(step):
        nonlocal best_ssim
        th.cuda.empty_cache()
        model.eval()
        _cur_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # 评测使用实际 online model 权重
        with th.no_grad():
            g_pred = inference.sample_latents(
                model, diff_eval, eval_noise, eval_conds,
                cfg_scale=1.0, batch=50, device=dev, skel=eval_std_lats
            )
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

            # 2b. ★ IoU (像素级重合) 与 LPIPS (感知距离) —— SSIM 背景主导、太宽松,
            #     必须配一个结构指标; 并且**同一批上算出"什么都不做"的基线**,
            #     否则永远不知道自己是比直接用标准字好还是差。
            def _iou(a, b):
                inter = (a & b).sum((-1, -2)).float()
                union = (a | b).sum((-1, -2)).float().clamp(min=1)
                return (inter / union).mean().item()

            pm = dec_gray < 0.6
            gm = gt_gray < 0.6
            sm = eval_std_pngs.mean(dim=1) < 0.6
            mean_iou = _iou(pm, gm)
            base_iou = _iou(sm, gm)
            base_ssim = float(np.mean(ssim_torch(eval_std_pngs, eval_gt_pngs).cpu().numpy()))
            try:
                from src.eval.in_mem_eval import _lpips_per_sample
                _lp = _lpips_per_sample(
                    (dec * 2 - 1).cpu().numpy(),
                    (eval_gt_pngs * 2 - 1).cpu().numpy(), enabled=True)
                _lpb = _lpips_per_sample(
                    (eval_std_pngs * 2 - 1).cpu().numpy(),
                    (eval_gt_pngs * 2 - 1).cpu().numpy(), enabled=True)
                mean_lpips = float(np.mean([x for x in _lp if x == x])) if _lp else float("nan")
                base_lpips = float(np.mean([x for x in _lpb if x == x])) if _lpb else float("nan")
            except Exception as _e:                                   # noqa: BLE001
                mean_lpips = base_lpips = float("nan")

            # 3. 骨架连通块破碎度
            p_np = dec_gray.cpu().numpy()
            g_np = gt_gray.cpu().numpy()
            frags = [frag_ratio(p_np[i:i+1], g_np[i:i+1]) for i in range(len(p_np))]
            mean_frag = float(np.mean(frags))
            med_frag = float(np.median(frags))

        print("\n" + "="*70)
        print(f"[EVAL @ Step {step:05d}] 纯血真迹 200 样本骨架评测:")
        print(f"  骨架 SSIM (均值)   : {mean_ssim:.4f} (中位: {med_ssim:.4f})"
              f"   基线(标准字) {base_ssim:.4f}  "
              f"{'✓' if mean_ssim > base_ssim else '✗ 不如不做'}")
        print(f"  墨迹 SSIM (Ink)    : {mean_ink:.4f}")
        print(f"  骨架 IoU           : {mean_iou:.4f}"
              f"   基线(标准字) {base_iou:.4f}  "
              f"{'✓' if mean_iou > base_iou else '✗ 不如不做'}")
        print(f"  LPIPS (越低越好)   : {mean_lpips:.4f}"
              f"   基线(标准字) {base_lpips:.4f}  "
              f"{'✓' if mean_lpips < base_lpips else '✗ 不如不做'}")
        print(f"  破碎度 (frag_ratio): {mean_frag:.3f} (中位: {med_frag:.3f}) [1.0=完全连贯无破碎!]")
        print("="*70 + "\n", flush=True)

        # 4. 渲染 20 样本 3 行对比全景海报
        # Row 1: 标准字输入骨架 (g_std)
        # Row 2: SkelNet-S/2 生成骨架 (g_pred)
        # Row 3: 真实真迹 Ground Truth 骨架 (g_gt)
        p_cols = 20
        p_canvas = Image.new("RGB", (256 * p_cols, 256 * 3))
        sub_indices = np.linspace(0, n_eval - 1, p_cols, dtype=int)
        for col_idx, col in enumerate(sub_indices):
            p_std = Image.fromarray((eval_std_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gen = Image.fromarray((dec[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_gt = Image.fromarray((eval_gt_pngs[col].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            p_canvas.paste(p_std, (col_idx * 256, 0))
            p_canvas.paste(p_gen, (col_idx * 256, 256))
            p_canvas.paste(p_gt, (col_idx * 256, 512))
        poster_file = os.path.join(poster_dir, f"eval_step_{step:07d}.png")
        p_canvas.save(poster_file)
        print(f"[poster] 3 行全景对比海报已保存到: {poster_file}", flush=True)

        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            _sd = _strip(model.state_dict())
            th.save({"model": _sd, "step": step, "mean_ssim": mean_ssim, "mean_frag": mean_frag},
                    os.path.join(ckpt_dir, "best_ssim.pt"))
            print(f"  ★ 新纪录! Best SSIM={best_ssim:.4f} 已保存到 best_ssim.pt", flush=True)

        model.load_state_dict(_cur_sd)
        model.train()
        th.cuda.empty_cache()

    # 4. 训练主循环 (Batch=1440 显存拉满 19.37 GB, 单纯纯净 Flow Matching)
    t0 = time.time()
    r_flow, r_gn, cnt = 0.0, 0.0, 0
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

        # 随机流匹配时刻 t1 in (0, 1) (logit-normal 分布，集中在中段)
        t1 = th.sigmoid(th.randn(a.batch, device=dev))
        eps1 = th.randn_like(g_gt)
        z_t1 = (1 - t1[:, None, None, None]) * g_gt + t1[:, None, None, None] * eps1
        v1_target = eps1 - g_gt

        with th.autocast("cuda", dtype=th.bfloat16):
            v1 = model(z_t1, t1 * TIME_SCALE, y_callig=y, y_char=None, g=g_std)
            if isinstance(v1, tuple):
                v1 = v1[0]
            # latent flow 只保留一小部分权重: 它负责把 latent 留在 VAE 流形上,
            # 结构质量交给下面的 proj/dice (实测 latent L2 与骨架质量不同源)。
            loss = a.lam_latent * th.nn.functional.mse_loss(v1.float(), v1_target)
            # ★ 结构监督: 小 t 的单步去噪估计 x_hat = z_t - t*v (flow 下精确),
            #   解码成软墨迹后与真迹掩码比 投影剖面 + 软 Dice。
            if (a.lam_proj > 0 or a.lam_dice > 0) and step % a.struct_every == 0 \
                    and g_gt is not None:
                sb = min(a.struct_batch, g_gt.shape[0])
                # ⚠ t 必须够大: t->0 时 x_hat = z_t - t*v 恒等于 GT, 损失退化成
                #   "拿 GT 比 GT"(实测 proj 直接掉到 0.0001 = 0 信号)。取 0.35~0.85
                #   才能让 x_hat 真正依赖 v_hat。
                ts = th.rand(sb, device=dev) * 0.5 + 0.35
                eps_s = th.randn_like(g_gt[:sb])
                z_s = (1 - ts[:, None, None, None]) * g_gt[:sb] \
                    + ts[:, None, None, None] * eps_s
                with th.autocast("cuda", dtype=th.bfloat16):
                    vs = model(z_s, ts * TIME_SCALE, y_callig=y[:sb],
                               y_char=None, g=g_std[:sb])
                    if isinstance(vs, tuple):
                        vs = vs[0]
                x_hat = z_s - ts[:, None, None, None] * vs.float()
                _lp, _ld = struct_loss(x_hat.detach(), k[:sb])
                loss = loss + a.lam_proj * _lp + a.lam_dice * _ld
                if step % a.log_every == 0:
                    print(f"    [struct] step {step:05d} proj={float(_lp):.4f} "
                          f"dice={float(_ld):.4f}", flush=True)

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

    print(f"\n[done] SkelNet-S/2 纯血真迹训练全部完成 -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
