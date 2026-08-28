#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""auto_eval_gpu.py — GPU 推理 + CPU 指标计算的自动评测进程。

与 train.py 解耦: 轮询 ckpt 目录, 发现新 checkpoint (带 .done 标记) 即评测:
  1) GPU: 加载 EMA 权重 → DDIM 50步采样 → VAE decode → 472 张图
  2) CPU: 保存全部 472 张 pred+GT 到磁盘 (每 ckpt 一个子目录)
  3) CPU: 计算 MSE / SSIM / Skel-IoU (逐图)
  4) 写 eval_auto_{step}.json (含全部指标)
  5) show5/seen5 展示图落盘 (与 pull_monitor 兼容)

与训练共享 GPU: 训练占 ~20G, eval 用 ~2-3G (batch=4).
"""
import argparse, glob, json, os, sys, time, datetime, traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# ── skel iou ──
try:
    from skimage.morphology import skeletonize
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    from scipy.ndimage import binary_erosion, generate_binary_structure

    def _skeletonize_fallback(binary):
        """简易骨架化: 迭代腐蚀直到宽度=1."""
        skel = np.zeros_like(binary)
        img = binary.copy()
        struct = generate_binary_structure(2, 2)
        while img.any():
            eroded = binary_erosion(img, structure=struct)
            skel |= img & ~eroded
            img = eroded
        return skel


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ── 模型构建 (复刻 train.py) ──
def build_model(args, device):
    from src.model import DiT_2Cond_models
    vae_downscale = getattr(args, "vae_downscale", 4)
    latent_size = args.image_size // vae_downscale
    latent_channels = int(getattr(args, "latent_channels", 4))
    cond_mode = getattr(args, "cond_mode", "2cond")
    model_cls = DiT_2Cond_models[args.model]
    model = model_cls(
        input_size=latent_size,
        in_channels=latent_channels,
        num_calligraphers=getattr(args, "num_calligraphers", 1011),
        num_characters=getattr(args, "num_characters", 7765),
        condition_fusion=getattr(args, "condition_fusion", "factorized_add"),
        callig_embed_dim=int(getattr(args, "callig_embed_dim", 128)),
        char_embed_dim=int(getattr(args, "char_embed_dim", 512)),
        learn_sigma=True,
        cond_drop_all_prob=float(getattr(args, "cond_drop_all_prob", 0.05)),
        cond_drop_one_prob=float(getattr(args, "cond_drop_one_prob", 0.25)),
        skel_head_enabled=getattr(args, "w_skel_head", 0) > 0,
        use_glyph_cond=getattr(args, "w_glyph_cond", False),
        glyph_scale_init=float(getattr(args, "glyph_scale_init", 0.4)),
    ).to(device).eval()
    return model


def load_vae(args, device):
    from diffusers.models import AutoencoderKL
    path = getattr(args, "vae_path", None)
    if path and os.path.exists(path):
        log(f"[vae] loading {path}")
        return AutoencoderKL.from_pretrained(path).to(device).eval()
    log(f"[vae] loading stabilityai/sd-vae-ft-{args.vae}")
    return AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()


def load_ckpt_weights(model, ckpt_path, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if use_ema and "ema" in ckpt:
        weights = ckpt["ema"]
        log(f"[model] loaded weights from ema ({os.path.basename(ckpt_path)})")
    elif "model" in ckpt:
        weights = ckpt["model"]
        log(f"[model] loaded weights from model ({os.path.basename(ckpt_path)})")
    else:
        weights = ckpt
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        log(f"[model] missing keys: {len(missing)}")
    if unexpected:
        log(f"[model] unexpected keys: {len(unexpected)}")
    del ckpt
    return model


# ── 数据缓存 ──
def build_eval_cache(csv_path, data_dir, image_size, n, cond_mode="2cond",
                     use_glyph_cond=False):
    from src.utils import MCCDDataset
    from torch.utils.data import DataLoader
    ds = MCCDDataset(csv_file=csv_path, root_dir=data_dir,
                     image_size=image_size, load_canny=False, load_skel=False,
                     use_glyph_cond=use_glyph_cond)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    conds, gts, g_all = [], [], []
    for b in loader:
        if len(conds) >= n:
            break
        for i in range(b["y_callig"].shape[0]):
            if cond_mode == "2cond":
                conds.append((b["y_callig"][i].item(), -1, b["y_char"][i].item()))
            else:
                conds.append((b["y_callig"][i].item(), b["y_script"][i].item(), b["y_char"][i].item()))
        gts.append(b["image"].cpu())
        if "g" in b and b["g"].numel() > 0:
            g_all.append(b["g"].cpu())
    gts = torch.cat(gts, dim=0)[:n]
    g_ret = torch.cat(g_all, dim=0)[:n] if g_all else None
    log(f"[cache] {csv_path} -> {len(conds)} samples")
    return {"conds": conds[:n], "gts": gts, "gs": g_ret}


# ── GPU DDIM 采样 ──
def gpu_sample_batch(model, vae, conds, gts, device, steps, cfg, seed,
                     cond_mode, latent_channels, latent_spatial, scaling_factor,
                     batch_size=4, glyph_init_mix=0.0, gs_batch=None):
    """GPU 上做 DDIM 采样 + VAE decode, 返回 CPU 上的 (decoded, gts) numpy."""
    from src.loss import create_diffusion
    ddim = create_diffusion(str(steps))
    decoded_all = []
    gt_all = []
    n = len(conds)
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            bs = j - i
            noise = torch.randn(bs, latent_channels, latent_spatial, latent_spatial, device=device)
            if glyph_init_mix < 1.0 and gs_batch is not None and gs_batch[i:j].shape[0] == bs:
                if glyph_init_mix <= 0.0:
                    z = gs_batch[i:j].to(device).clone()
                else:
                    z = glyph_init_mix * noise + (1.0 - glyph_init_mix) * gs_batch[i:j].to(device)
            else:
                z = noise
            if cond_mode == "2cond":
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[i:j]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[i:j]], device=device),
                    cfg_scale=cfg,
                )
            else:
                mk = dict(
                    y_callig=torch.tensor([c[0] for c in conds[i:j]], device=device),
                    y_script=torch.tensor([c[1] for c in conds[i:j]], device=device),
                    y_char=torch.tensor([c[2] for c in conds[i:j]], device=device),
                    cfg_scale=cfg,
                )
            if gs_batch is not None and gs_batch[i:j].shape[0] == bs:
                mk["g"] = gs_batch[i:j].to(device)
            samples = ddim.ddim_sample_loop(model.forward_with_cfg, z.shape, z,
                                            clip_denoised=False, model_kwargs=mk, device=device)
            dec = vae.decode(samples / scaling_factor).sample  # (bs,3,256,256) [-1,1]
            decoded_all.append(dec.cpu())
            gt_all.append(gts[i:j].cpu())
            del z, samples, dec
            torch.cuda.empty_cache()
    decoded_all = torch.cat(decoded_all, dim=0)  # (n,3,256,256)
    gt_all = torch.cat(gt_all, dim=0)
    return decoded_all, gt_all


# ── CPU 指标 ──
def _ssim_numpy(img1, img2, win=7):
    """SSIM per-channel average. img1/img2: (H,W,3) float [0,1]"""
    from scipy.ndimage import uniform_filter
    c1, c2 = 0.01**2, 0.03**2
    ssims = []
    for ch in range(3):
        x = img1[:, :, ch].astype(np.float64)
        y = img2[:, :, ch].astype(np.float64)
        mu_x = uniform_filter(x, size=win)
        mu_y = uniform_filter(y, size=win)
        mu_x2 = mu_x**2; mu_y2 = mu_y**2; mu_xy = mu_x * mu_y
        sigma_x2 = uniform_filter(x*x, size=win) - mu_x2
        sigma_y2 = uniform_filter(y*y, size=win) - mu_y2
        sigma_xy = uniform_filter(x*y, size=win) - mu_xy
        ssim_map = ((2*mu_xy+c1)*(2*sigma_xy+c2)) / ((mu_x2+mu_y2+c1)*(sigma_x2+sigma_y2+c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def _skel_iou(img1, img2, thresh=0.5):
    """Skeleton IoU. img1/img2: (H,W,3) float [0,1]."""
    # 二值化 (取灰度)
    g1 = img1.mean(axis=2)
    g2 = img2.mean(axis=2)
    b1 = g1 < thresh  # 书法: 黑色笔画
    b2 = g2 < thresh
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    if _HAS_SKIMAGE:
        s1 = skeletonize(b1)
        s2 = skeletonize(b2)
    else:
        s1 = _skeletonize_fallback(b1)
        s2 = _skeletonize_fallback(b2)
    inter = (s1 & s2).sum()
    union = (s1 | s2).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def compute_metrics(decoded, gts):
    """decoded, gts: (n,3,256,256) tensor [-1,1]. 返回逐图 metrics."""
    n = decoded.shape[0]
    decoded_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()  # (n,3,256,256) [0,1]
    gts_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    results = []
    for i in range(n):
        pred = decoded_np[i].transpose(1, 2, 0)  # (H,W,3)
        gt = gts_np[i].transpose(1, 2, 0)
        mse = float(np.mean((pred - gt)**2))
        ssim = _ssim_numpy(pred, gt)
        skel_iou = _skel_iou(pred, gt)
        results.append({"idx": i, "mse": mse, "ssim": ssim, "skel_iou": skel_iou})
    return results


def save_images(decoded, gts, conds, out_dir, step):
    """保存全部图片到 out_dir/stepXXXXXXX/."""
    step_tag = f"step{int(step):07d}"
    step_dir = os.path.join(out_dir, step_tag)
    os.makedirs(step_dir, exist_ok=True)
    decoded_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()
    gts_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    n = decoded.shape[0]
    for i in range(n):
        pred_img = (decoded_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        gt_img = (gts_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        Image.fromarray(pred_img).save(os.path.join(step_dir, f"sample{i}.png"))
        Image.fromarray(gt_img).save(os.path.join(step_dir, f"gt{i}.png"))
    # 元数据
    with open(os.path.join(step_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "n": n,
                   "conds": [list(c) for c in conds[:n]]}, f, ensure_ascii=False)
    return step_dir


# ── 单 ckpt 评测 ──
def eval_one_ckpt(model, vae, ckpt_path, ckpt_name, step, ckpt_dir,
                  eval_cache, show5_cache, seen5_cache, device, cfg):
    t0 = time.time()
    # 1) 加载权重
    model = load_ckpt_weights(model, ckpt_path, use_ema=True)

    # 2) GPU 采样 eval500
    log(f"[eval] === processing {ckpt_name} (step {step}) ===")
    decoded, gts = gpu_sample_batch(
        model, vae, eval_cache["conds"], eval_cache["gts"], device,
        steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
        cond_mode=cfg["cond_mode"],
        latent_channels=cfg["latent_channels"],
        latent_spatial=cfg["latent_spatial"],
        scaling_factor=cfg["scaling_factor"],
        batch_size=cfg["batch"],
        glyph_init_mix=cfg["glyph_init_mix"],
        gs_batch=eval_cache.get("gs"),
    )

    # 3) 保存全部 472 张图
    save_dir = os.path.join(ckpt_dir, "eval_samples")
    save_images(decoded, gts, eval_cache["conds"], save_dir, step)
    log(f"[eval] saved {decoded.shape[0]} images to {save_dir}/step{step:07d}/")

    # 4) CPU 计算 metrics
    metrics = compute_metrics(decoded, gts)
    mses = [m["mse"] for m in metrics]
    ssims = [m["ssim"] for m in metrics]
    skel_ious = [m["skel_iou"] for m in metrics]
    mse_mean = float(np.mean(mses))
    ssim_mean = float(np.mean(ssims))
    skel_iou_mean = float(np.mean(skel_ious))

    # 5) show5 (展示图)
    show5_dir = os.path.join(ckpt_dir, "eval_samples")
    show5_dec, show5_gt = gpu_sample_batch(
        model, vae, show5_cache["conds"], show5_cache["gts"], device,
        steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
        cond_mode=cfg["cond_mode"],
        latent_channels=cfg["latent_channels"],
        latent_spatial=cfg["latent_spatial"],
        scaling_factor=cfg["scaling_factor"],
        batch_size=16, glyph_init_mix=cfg["glyph_init_mix"],
        gs_batch=show5_cache.get("gs"),
    )
    save_images(show5_dec, show5_gt, show5_cache["conds"], show5_dir, step)
    # eval_latest.png (拼图)
    _save_latest_thumb(show5_dec, show5_gt, os.path.join(ckpt_dir, "eval_latest.png"))

    # 6) seen5
    seen5_dir = os.path.join(ckpt_dir, "seen_samples")
    seen5_dec, seen5_gt = gpu_sample_batch(
        model, vae, seen5_cache["conds"], seen5_cache["gts"], device,
        steps=cfg["steps"], cfg=cfg["cfg"], seed=cfg["seed"],
        cond_mode=cfg["cond_mode"],
        latent_channels=cfg["latent_channels"],
        latent_spatial=cfg["latent_spatial"],
        scaling_factor=cfg["scaling_factor"],
        batch_size=16, glyph_init_mix=cfg["glyph_init_mix"],
        gs_batch=seen5_cache.get("gs"),
    )
    save_images(seen5_dec, seen5_gt, seen5_cache["conds"], seen5_dir, step)

    elapsed = time.time() - t0
    log(f"[eval] step {step}: MSE={mse_mean:.5f} SSIM={ssim_mean:.4f} "
        f"SkelIoU={skel_iou_mean:.4f} ({elapsed:.0f}s)")

    # 7) 写 metrics JSON
    result = {
        "step": step, "mse": mse_mean, "ssim": ssim_mean,
        "skel_iou": skel_iou_mean,
        "mse_std": float(np.std(mses)),
        "ssim_std": float(np.std(ssims)),
        "skel_iou_std": float(np.std(skel_ious)),
        "ssim_min": float(np.min(ssims)),
        "skel_iou_min": float(np.min(skel_ious)),
        "n": len(metrics),
        "elapsed": elapsed,
    }
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
        json.dump(result, f, indent=2)

    del decoded, gts, show5_dec, show5_gt, seen5_dec, seen5_gt
    torch.cuda.empty_cache()
    return result


def _save_latest_thumb(decoded, gts, out_path):
    """保存 show5 拼图 (pred | gt 并排)."""
    n = decoded.shape[0]
    h, w = decoded.shape[2], decoded.shape[3]
    canvas = np.zeros((h * 2, w * n, 3), dtype=np.uint8)
    dec_np = ((decoded.clamp(-1, 1) + 1) / 2).numpy()
    gt_np = ((gts.clamp(-1, 1) + 1) / 2).numpy()
    for i in range(n):
        canvas[:h, i*w:(i+1)*w] = (dec_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        canvas[h:, i*w:(i+1)*w] = (gt_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    Image.fromarray(canvas).save(out_path)


# ── 轮询主循环 ──
def read_active_ckpt_dir(results_dir):
    marker = os.path.join(results_dir, "_active_ckpt_dir.txt")
    if not os.path.exists(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() or None


def load_state(ckpt_dir):
    sp = os.path.join(ckpt_dir, "gpu_eval_state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(ckpt_dir, state):
    sp = os.path.join(ckpt_dir, "gpu_eval_state.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_ckpt_args(ckpt_path):
    """从 ckpt 中读取 args (训练配置)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or ckpt.get("config")
    if args is None:
        raise ValueError(f"no args/config in {ckpt_path}")
    del ckpt
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--show5-csv", default=None)
    ap.add_argument("--seen5-csv", default=None)
    ap.add_argument("--eval-n", type=int, default=472)
    ap.add_argument("--batch-size", type=int, default=4, help="GPU eval batch size")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    os.chdir("/root/Workspace/xy/DiT")
    device = "cuda"
    log(f"[main] device={device}, eval_csv={args.eval_csv}, eval_n={args.eval_n}")

    # 找到 active ckpt dir
    ckpt_dir = None
    while ckpt_dir is None:
        ckpt_dir = read_active_ckpt_dir(args.results_dir)
        if ckpt_dir is None:
            log("[watch] no active ckpt dir yet, waiting...")
            time.sleep(30)
    ckpt_dir = os.path.join("/root/Workspace/xy/DiT", ckpt_dir) if not os.path.isabs(ckpt_dir) else ckpt_dir
    log(f"[watch] active ckpt dir: {ckpt_dir}")

    # 等第一个 ckpt 出现, 读取 args
    first_ckpt = None
    while first_ckpt is None:
        cks = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))
        if cks:
            first_ckpt = cks[0].replace(".done", "")
            break
        log("[watch] no checkpoints yet, waiting...")
        time.sleep(30)

    train_args = get_ckpt_args(first_ckpt)
    log(f"[main] loaded train args from {os.path.basename(first_ckpt)}")

    # 构建模型 + VAE
    model = build_model(train_args, device)
    vae = load_vae(train_args, device)

    # 注入 DINO (如有)
    dino_emb = getattr(train_args, "char_dino_embeddings", None)
    dino_idx = getattr(train_args, "char_dino_index", None)
    if dino_emb and dino_idx and os.path.exists(dino_emb):
        _inject_dino(model, train_args)
        log("[dino] injected glyph embeddings")

    # 构建缓存
    cond_mode = getattr(train_args, "cond_mode", "2cond")
    use_glyph_cond = getattr(train_args, "w_glyph_cond", False)
    eval_cache = build_eval_cache(args.eval_csv, getattr(train_args, "data_dir", ""),
                                  train_args.image_size, args.eval_n, cond_mode, use_glyph_cond)
    show5_cache = build_eval_cache(args.show5_csv, getattr(train_args, "data_dir", ""),
                                   train_args.image_size, 100, cond_mode, use_glyph_cond) if args.show5_csv else None
    seen5_cache = build_eval_cache(args.seen5_csv, getattr(train_args, "data_dir", ""),
                                  train_args.image_size, 100, cond_mode, use_glyph_cond) if args.seen5_csv else None

    # eval 配置
    cfg = {
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "batch": args.batch_size,
        "cond_mode": cond_mode,
        "glyph_init_mix": float(getattr(train_args, "glyph_init_mix", 0.0)),
        "latent_channels": int(getattr(train_args, "latent_channels", 4)),
        "latent_spatial": int(train_args.image_size) // int(getattr(train_args, "vae_downscale", 8)),
        "scaling_factor": float(getattr(train_args, "vae_scaling_factor", 0.18215)),
    }
    log(f"[main] eval config: {cfg}")

    # 轮询
    state = load_state(ckpt_dir)
    done_ckpts = set(state.get("done", []))

    while True:
        # 找所有 .pt.done
        all_done = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.done")))
        pending = []
        for d in all_done:
            ckpt_path = d.replace(".done", "")
            ckpt_name = os.path.basename(ckpt_path)
            if ckpt_name not in done_ckpts:
                pending.append(ckpt_path)
        if pending:
            log(f"[scan] {len(all_done)} ckpts total, {len(done_ckpts)} done, {len(pending)} pending")
        else:
            log(f"[scan] {len(all_done)} ckpts, {len(done_ckpts)} done, 0 pending — waiting")

        for ckpt_path in pending:
            ckpt_name = os.path.basename(ckpt_path)
            # 从文件名提取 step
            try:
                step = int(ckpt_name.replace(".pt", "").lstrip("0") or "0")
            except ValueError:
                step = 0
            try:
                eval_one_ckpt(model, vae, ckpt_path, ckpt_name, step, ckpt_dir,
                              eval_cache, show5_cache, seen5_cache, device, cfg)
                done_ckpts.add(ckpt_name)
                state["done"] = sorted(done_ckpts)
                save_state(ckpt_dir, state)
            except Exception as e:
                log(f"[eval] step {step}: FAILED: {e}")
                log(traceback.format_exc())
                done_ckpts.add(ckpt_name)  # 不重试
                state["done"] = sorted(done_ckpts)
                save_state(ckpt_dir, state)

        if args.once:
            break
        time.sleep(args.interval)


def _inject_dino(model, args):
    """注入 DINO glyph embeddings (复刻 train.py 逻辑)."""
    emb_path = getattr(args, "char_dino_embeddings", None)
    idx_path = getattr(args, "char_dino_index", None)
    if not emb_path or not idx_path:
        return
    emb = np.load(emb_path)  # (n_glyphs, 768)
    with open(idx_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    glyphs = idx_data.get("glyphs", idx_data)  # [[script_id, char_id], ...]
    table = model.y_char_embedder.embedding_table.weight
    NUM_CH = 7026
    injected = 0
    with torch.no_grad():
        for gi, (sid, cid) in enumerate(glyphs):
            gid = int(sid) * NUM_CH + int(cid)
            if 0 <= gid < table.shape[0] and gi < emb.shape[0]:
                e = emb[gi]
                e = e / (np.linalg.norm(e) + 1e-8)
                table.data[gid] = torch.from_numpy(e).float()
                injected += 1
    log(f"[dino-init] injected {injected} glyph embeddings")


if __name__ == "__main__":
    main()
