# -*- coding: utf-8 -*-
"""
auto_eval_ctrl.py — ControlNet 专用 CPU 评测进程 (适配自 auto_eval_cpu.py).

轮询 ctrl 训练实验的 ckpt 目录, 发现新 checkpoint 即评测:
  1) base:  无 skel 条件 → DDIM 自由采样 → MSE/SSIM vs GT
  2) ctrl:  有 GT skel 条件 → DDIM 自由采样 → MSE/SSIM vs GT
  → <ckpt_dir>/eval_auto_{step}.json (含 base/ctrl 两组指标)

ControlNet ckpt 只含 ctrl_encoder 权重, 主模型 (195k) 固定。
模型构建: 加载 195k 主模型 + ctrl ckpt → ControlNetDiT。

用法:
  python auto_eval_ctrl.py --results-dir 5script/results/ctrl_skel [--interval 30]
"""
import argparse
import glob
import json
import os
import sys
import time
import datetime

import numpy as np
import torch

# 确保能 import controlnet_dit (在 tools/controlnet/) 和 src/ 模块
_ctrl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "controlnet")
if _ctrl_dir not in sys.path:
    sys.path.insert(0, _ctrl_dir)
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MAIN_CKPT = "5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt"
DEFAULT_EVAL_CSV = "5script/eval100_top6.csv"
DEFAULT_VAE = "pretrained_models/sd-vae-ft-ema"
SKEL_ROOT = "final_skeleton_d3"
IMG_ROOT = "final_imgs_256"


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模型构建 (ControlNet: frozen main 195k + ctrl ckpt, 或 from-scratch main+ctrl)
# ---------------------------------------------------------------------------
def build_main_model(device="cpu", from_scratch=False, main_ckpt=None):
    """构建主模型. from_scratch=True 时不加载 195k (随机初始化, 等待 ckpt 覆盖)."""
    from controlnet_dit import load_main_model
    ckpt_path = None if from_scratch else (main_ckpt or DEFAULT_MAIN_CKPT)
    model = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=ckpt_path, device=device,
        num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=256, cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False)
    model.eval()
    return model


def build_ctrl(main_model, device="cpu", train_ctrl_only=True):
    from controlnet_dit import ControlNetDiT
    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=train_ctrl_only).to(device)
    ctrl.eval()
    return ctrl


def load_ctrl_weights(ctrl, ckpt, base):
    """Load ctrl_encoder (+ main.* if present) weights from ckpt (ema or raw)."""
    sd = ckpt.get("ema")
    src = "ema"
    if sd is None:
        sd = ckpt.get("ctrl")
        src = "ctrl"
    if sd is None:
        log(f"[ctrl] {base}: no ema/ctrl weights, using zero-init")
        return src

    # ctrl_encoder 权重
    ctrl_keys = {k: v for k, v in sd.items() if k.startswith("ctrl_encoder")}
    missing, unexpected = ctrl.load_state_dict(ctrl_keys, strict=False)
    log(f"[ctrl] loaded {src} from {base} ({len(ctrl_keys)} keys, "
        f"missing={len(missing)}, unexpected={len(unexpected)})")

    # from-scratch: 还需要加载 main.* 权重
    if "model" in ckpt and ckpt["model"]:
        main_sd = ckpt["model"]
        main_keys = {k: v for k, v in main_sd.items() if k.startswith("main.")}
        if main_keys:
            m_missing, m_unexpected = ctrl.load_state_dict(main_keys, strict=False)
            log(f"[main] loaded model from {base} ({len(main_keys)} keys, "
                f"missing={len(m_missing)}, unexpected={len(m_unexpected)})")
    # ema_model 优先
    if "ema_model" in ckpt and ckpt["ema_model"]:
        ema_main_sd = ckpt["ema_model"]
        ema_main_keys = {k: v for k, v in ema_main_sd.items() if k.startswith("main.")}
        if ema_main_keys:
            m_missing, m_unexpected = ctrl.load_state_dict(ema_main_keys, strict=False)
            log(f"[main] loaded ema_model from {base} ({len(ema_main_keys)} keys, "
                f"missing={len(m_missing)}, unexpected={len(m_unexpected)})")
    return src


def load_vae(device="cpu"):
    from diffusers.models import AutoencoderKL
    log(f"[vae] loading {DEFAULT_VAE}")
    return AutoencoderKL.from_pretrained(DEFAULT_VAE).to(device).eval()


# ---------------------------------------------------------------------------
# 数据缓存 (eval 样本: 条件 + GT 图 + skel)
# ---------------------------------------------------------------------------
def build_cache(eval_csv, n=100):
    """缓存 n 个 eval 样本: conds + gts + skels. 手动遍历 + resize 避免 collate 问题."""
    from latent_dataset import MCCDLatentDataset
    import torch.nn.functional as Fn
    ds = MCCDLatentDataset(
        csv_file=eval_csv, latent_shards_dir="final_latents",
        img_root=IMG_ROOT, skel_root=SKEL_ROOT,
        image_size=256, load_skel=True, load_image=True,
        is_train=False, preload=False, structure_size=256)
    conds, gts, skels = [], [], []
    for idx in range(min(n, len(ds))):
        s = ds[idx]
        img = s['image']
        if img.shape[-1] != 256:
            img = Fn.interpolate(img.unsqueeze(0), size=256, mode="bilinear",
                                align_corners=False).squeeze(0)
        skel = s['skeleton']
        if skel.shape[-1] != 256:
            skel = Fn.interpolate(skel.unsqueeze(0), size=256, mode="area").squeeze(0)
        conds.append((s['y_callig'].item(), -1, s['y_char'].item()))
        gts.append(img.unsqueeze(0))
        skels.append(skel.unsqueeze(0))
    gts = torch.cat(gts, dim=0)[:n]
    skels = torch.cat(skels, dim=0)[:n]
    log(f"[cache] {eval_csv} -> {len(conds)} samples")
    return {"conds": conds[:n], "gts": gts, "skels": skels}


# ---------------------------------------------------------------------------
# 指标 (批量, numpy/scipy CPU; 与 eval_metrics_daemon.py 同口径)
# ---------------------------------------------------------------------------
def _ssim_batch(pred, gt, win=11, data_range=1.0):
    """SSIM for (N,3,H,W) float32 [0,1] arrays. 批量: 所有图×通道一次 uniform_filter."""
    from scipy.ndimage import uniform_filter
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    n = pred.shape[0]
    # (N,3,H,W) -> (N*3,H,W), 一次过滤所有通道
    x = pred.reshape(-1, pred.shape[2], pred.shape[3]).astype(np.float64)
    y = gt.reshape(-1, gt.shape[2], gt.shape[3]).astype(np.float64)
    mu_x = uniform_filter(x, size=win)
    mu_y = uniform_filter(y, size=win)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = uniform_filter(x * x, size=win) - mu_x2
    sy2 = uniform_filter(y * y, size=win) - mu_y2
    sxy = uniform_filter(x * y, size=win) - mu_xy
    m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
    # 每图 3 通道平均
    per_img = m.reshape(n, 3, -1).mean(axis=(1, 2))
    return float(per_img.mean())


# ---------------------------------------------------------------------------
# LPIPS + skel_iou (与 eval_metrics_daemon.py 同口径, CPU)
# ---------------------------------------------------------------------------
_lpips_fn = None
_lpips_loaded = False

def _get_lpips():
    """Lazily load LPIPS model on CPU. Returns None if unavailable."""
    global _lpips_fn, _lpips_loaded
    if _lpips_loaded:
        return _lpips_fn
    _lpips_loaded = True
    try:
        import lpips
        _lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_fn.eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
        log("LPIPS model loaded (vgg, CPU)")
    except Exception as e:
        log(f"LPIPS unavailable ({e}), will skip lpips metric")
        _lpips_fn = None
    return _lpips_fn


def _lpips_batch(pred_t, gt_t, lpips_fn):
    """pred_t/gt_t: (N,3,H,W) [-1,1] tensors. 一次批量 forward. Returns float."""
    if lpips_fn is None:
        return None
    p = pred_t.float().cpu()
    g = gt_t.float().cpu()
    with torch.no_grad():
        return float(lpips_fn(p, g).mean().item())


def _skel_iou_batch(pred_np, gt_np, thresh=0.5):
    """pred_np/gt_np: (N,3,H,W) float32 [0,1]. 批量 Skeleton IoU (白底黑字, 笔画<thresh)."""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skeletonize(binary):
            skel = np.zeros_like(binary)
            img = binary.copy()
            struct = generate_binary_structure(2, 2)
            while img.any():
                eroded = binary_erosion(img, structure=struct)
                skel |= img & ~eroded
                img = eroded
            return skel

    n = pred_np.shape[0]
    g1 = pred_np.mean(axis=3)  # (N,H,W)
    g2 = gt_np.mean(axis=3)
    b1 = g1 < thresh
    b2 = g2 < thresh
    inter_sum = 0.0
    union_sum = 0.0
    for k in range(n):
        if not b1[k].any() and not b2[k].any():
            inter_sum += 1.0
            union_sum += 1.0
            continue
        if not b1[k].any() or not b2[k].any():
            union_sum += 1.0  # IoU=0, union 至少为 1 防除零 (与 daemon 口径一致)
            continue
        s1 = skeletonize(b1[k])
        s2 = skeletonize(b2[k])
        inter_sum += float((s1 & s2).sum())
        union_sum += float((s1 | s2).sum())
    return inter_sum / union_sum if union_sum > 0 else 1.0


# ---------------------------------------------------------------------------
# 单 ckpt 评测: base vs ctrl, 自由采样 DDIM
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_one_step(model, vae, diffusion, device, cache, n=100, steps=50,
                  cfg=4.0, seed=0, batch=16, use_skel=False):
    """自由采样 → VAE decode → 批量 MSE/SSIM/LPIPS/skel_iou vs GT.
    use_skel=True 时传入 GT skel 作为 cond; False 时 cond=None.
    Returns (mse, ssim, lpips, skel_iou); lpips=None if unavailable.
    """
    lpips_fn = _get_lpips()
    conds = cache["conds"][:n]
    gts = cache["gts"][:n].to(device)
    skels = cache["skels"][:n].to(device) if "skels" in cache else None
    decs, gts_all = [], []
    mse_sum, cnt = 0.0, 0
    torch.manual_seed(seed)
    for i in range(0, n, batch):
        j = min(i + batch, n)
        z = torch.randn(j - i, 4, 32, 32, device=device)
        yc = torch.tensor([c[0] for c in conds[i:j]], device=device, dtype=torch.long)
        yh = torch.tensor([c[2] for c in conds[i:j]], device=device, dtype=torch.long)
        mk = dict(y_callig=yc, y_char=yh, cfg_scale=cfg)
        if use_skel and skels is not None:
            mk["cond"] = skels[i:j].float()
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z,
            clip_denoised=False, model_kwargs=mk, device=device)
        dec = vae.decode(samples / 0.18215).sample   # [-1,1]
        gt = gts[i:j]
        mse_sum += torch.nn.functional.mse_loss(dec, gt).item() * (j - i)
        decs.append(dec.cpu())
        gts_all.append(gt.cpu())
        cnt += (j - i)

    dec_cat = torch.cat(decs, dim=0)        # (N,3,256,256) [-1,1]
    gt_cat = torch.cat(gts_all, dim=0)      # (N,3,256,256) [-1,1]
    dec01 = ((dec_cat + 1) / 2).clamp(0, 1).numpy()
    gt01 = ((gt_cat + 1) / 2).clamp(0, 1).numpy()

    mse = mse_sum / cnt
    ssim = _ssim_batch(dec01, gt01)
    lpips = _lpips_batch(dec_cat, gt_cat, lpips_fn)
    skel_iou = _skel_iou_batch(dec01, gt01)
    return mse, ssim, lpips, skel_iou


def eval_ckpt(ctrl, vae, diffusion, device, cache, ckpt_dir, step, cfg_params):
    """Evaluate one ckpt: base (no skel) + ctrl (with skel)."""
    n = len(cache["conds"])
    t0 = time.time()

    # 1) base: 无 skel (退化为主模型)
    log(f"[eval] step {step}: base (no skel) ...")
    mse_base, ssim_base, lpips_base, skel_base = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        steps=cfg_params["steps"], cfg=cfg_params["cfg"],
        seed=cfg_params["seed"], batch=cfg_params["batch"], use_skel=False)
    log(f"[eval] step {step}: base  MSE={mse_base:.5f} SSIM={ssim_base:.4f}"
        f" LPIPS={lpips_base:.4f}" if lpips_base is not None
        else f"[eval] step {step}: base  MSE={mse_base:.5f} SSIM={ssim_base:.4f} LPIPS=n/a")
    log(f"[eval] step {step}: base  SkelIoU={skel_base:.4f}")

    # 2) ctrl: 有 GT skel
    log(f"[eval] step {step}: ctrl (GT skel) ...")
    mse_ctrl, ssim_ctrl, lpips_ctrl, skel_ctrl = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        steps=cfg_params["steps"], cfg=cfg_params["cfg"],
        seed=cfg_params["seed"], batch=cfg_params["batch"], use_skel=True)
    log(f"[eval] step {step}: ctrl  MSE={mse_ctrl:.5f} SSIM={ssim_ctrl:.4f}"
        f" LPIPS={lpips_ctrl:.4f}" if lpips_ctrl is not None
        else f"[eval] step {step}: ctrl  MSE={mse_ctrl:.5f} SSIM={ssim_ctrl:.4f} LPIPS=n/a")
    log(f"[eval] step {step}: ctrl  SkelIoU={skel_ctrl:.4f}")

    result = {
        "step": step,
        "mse_base": mse_base, "ssim_base": ssim_base,
        "skel_iou_base": skel_base,
        "mse_ctrl": mse_ctrl, "ssim_ctrl": ssim_ctrl,
        "skel_iou_ctrl": skel_ctrl,
        "delta_mse": mse_ctrl - mse_base,
        "delta_ssim": ssim_ctrl - ssim_base,
        "delta_skel_iou": skel_ctrl - skel_base,
    }
    if lpips_base is not None:
        result["lpips_base"] = lpips_base
        result["delta_lpips"] = lpips_ctrl - lpips_base
    if lpips_ctrl is not None:
        result["lpips_ctrl"] = lpips_ctrl
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[eval] step {step}: done ({time.time()-t0:.0f}s) "
        f"ΔMSE={result['delta_mse']:+.5f} ΔSSIM={result['delta_ssim']:+.4f}"
        f" ΔSkelIoU={result['delta_skel_iou']:+.4f}")
    return result


# ---------------------------------------------------------------------------
# 轮询主循环 (适配自 auto_eval_cpu.py)
# ---------------------------------------------------------------------------
def read_active_ckpt_dir(results_dir):
    marker = os.path.join(results_dir, "_active_ckpt_dir.txt")
    if not os.path.exists(marker):
        return None
    with open(marker, encoding="utf-8") as f:
        return f.read().strip() or None


def load_state(ckpt_dir):
    sp = os.path.join(ckpt_dir, "cpu_eval_state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(ckpt_dir, state):
    sp = os.path.join(ckpt_dir, "cpu_eval_state.json")
    with open(sp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="5script/results/ctrl_skel",
                    help="ctrl 训练 results 目录 (train 写 _active_ckpt_dir.txt)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="直接指定 ckpt 目录 (优先于轮询)")
    ap.add_argument("--interval", type=int, default=30, help="轮询间隔秒")
    ap.add_argument("--once", action="store_true", help="只处理当前所有新 ckpt 一次后退出")
    ap.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="eval 设备 (cuda 用于 GPU 批量评测)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="from-scratch 模式: 不加载 195k 主模型 (ckpt 自带 main 权重)")
    ap.add_argument("--main-ckpt", default=None,
                    help="warm-start 主模型 ckpt (默认 195k)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"[init] device={device}, eval_csv={args.eval_csv}, n={args.eval_n}, "
        f"from_scratch={args.from_scratch}")

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # 加载共享组件 (只一次): 主模型 + ctrl shell + VAE + cache + diffusion
    log("[init] building main model ...")
    main_model = build_main_model(device, from_scratch=args.from_scratch,
                                   main_ckpt=args.main_ckpt)
    log("[init] building ctrl shell ...")
    ctrl = build_ctrl(main_model, device, train_ctrl_only=not args.from_scratch)
    log("[init] loading VAE ...")
    vae = load_vae(device)
    log("[init] building eval cache ...")
    cache = build_cache(args.eval_csv, n=args.eval_n)
    if device.type == "cuda":
        cache["gts"] = cache["gts"].to(device)
        cache["skels"] = cache["skels"].to(device)
    from diffusion import create_diffusion
    diffusion = create_diffusion(str(args.steps))

    cfg_params = {"steps": args.steps, "cfg": args.cfg,
                  "seed": args.seed, "batch": args.batch}

    last_ckpt_dir = None
    state = {}

    while True:
        ckpt_dir = args.ckpt_dir or read_active_ckpt_dir(results_dir)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            log(f"[wait] no active ckpt dir ({results_dir}) ...")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        if ckpt_dir != last_ckpt_dir:
            log(f"[watch] active ckpt dir: {ckpt_dir}")
            last_ckpt_dir = ckpt_dir
            state = load_state(ckpt_dir)

        done_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not done_files:
            log(f"[wait] no checkpoints in {ckpt_dir}")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        for pt in done_files:
            base = os.path.basename(pt)
            if base in state:
                continue
            if not os.path.exists(pt + ".done"):
                log(f"[skip] {base}: .done marker missing")
                continue
            try:
                ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            except Exception as e:
                log(f"[warn] load {base} failed: {e}")
                continue
            step = int(ckpt.get("train_steps", 0) or 0)
            log(f"[eval] === processing {base} (step {step}) ===")

            # 只换 ctrl 权重, 不重建主模型
            load_ctrl_weights(ctrl, ckpt, base)

            try:
                res = eval_ckpt(ctrl, vae, diffusion, device, cache,
                                ckpt_dir, step, cfg_params)
                state[base] = {"step": step, "ok": True,
                               "mse_base": res["mse_base"],
                               "mse_ctrl": res["mse_ctrl"],
                               "ts": datetime.datetime.now().isoformat()}
                save_state(ckpt_dir, state)
            except Exception as e:
                import traceback
                log(f"[error] eval {base} failed: {e}")
                traceback.print_exc()
                state[base] = {"step": step, "error": str(e)}
                save_state(ckpt_dir, state)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
