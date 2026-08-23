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
IMG_ROOT = "final_images"


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 模型构建 (ControlNet: frozen main 195k + ctrl ckpt)
# ---------------------------------------------------------------------------
def build_main_model(device="cpu"):
    from controlnet_dit import load_main_model
    model = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=DEFAULT_MAIN_CKPT, device=device,
        num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=256, cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
        use_checkpoint=False)
    model.eval()
    return model


def build_ctrl(main_model, device="cpu"):
    from controlnet_dit import ControlNetDiT
    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True).to(device)
    ctrl.eval()
    return ctrl


def load_ctrl_weights(ctrl, ckpt, base):
    """Load ctrl_encoder weights from ckpt (ema or ctrl)."""
    sd = ckpt.get("ema")
    src = "ema"
    if sd is None:
        sd = ckpt.get("ctrl")
        src = "ctrl"
    if sd is None:
        log(f"[ctrl] {base}: no ema/ctrl weights, using zero-init")
        return src
    ctrl_keys = {k: v for k, v in sd.items() if k.startswith("ctrl_encoder")}
    missing, unexpected = ctrl.load_state_dict(ctrl_keys, strict=False)
    log(f"[ctrl] loaded {src} from {base} ({len(ctrl_keys)} keys, "
        f"missing={len(missing)}, unexpected={len(unexpected)})")
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
# SSIM (from eval_auto)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    g = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return (g.reshape(1, 1, window_size, 1) @ g.reshape(1, 1, 1, window_size))


def _ssim(x, y, data_range=1.0, window_size=11, win=None):
    import torch.nn.functional as F
    if x.shape[1] == 3:
        return sum(_ssim(x[:, i:i + 1], y[:, i:i + 1], data_range, window_size, win)
                   for i in range(3)) / 3
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, win, padding=window_size // 2)
    mu_y = F.conv2d(y, win, padding=window_size // 2)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = F.conv2d(x * x, win, padding=window_size // 2) - mu_x2
    sy2 = F.conv2d(y * y, win, padding=window_size // 2) - mu_y2
    sxy = F.conv2d(x * y, win, padding=window_size // 2) - mu_xy
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return float(m.mean().item())


# ---------------------------------------------------------------------------
# 单 ckpt 评测: base vs ctrl, 自由采样 DDIM
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_one_step(model, vae, diffusion, device, cache, n=100, steps=50,
                  cfg=4.0, seed=0, batch=16, use_skel=False):
    """自由采样 → VAE decode → MSE/SSIM vs GT.
    use_skel=True 时传入 GT skel 作为 cond; False 时 cond=None.
    """
    win = _gaussian_window(11, 1.5, device)
    conds = cache["conds"][:n]
    gts = cache["gts"][:n].to(device)
    skels = cache["skels"][:n].to(device) if "skels" in cache else None
    mse_sum, ssim_sum, cnt = 0.0, 0.0, 0
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
        dec = vae.decode(samples / 0.18215).sample
        gt = gts[i:j]
        mse_sum += torch.nn.functional.mse_loss(dec, gt).item() * (j - i)
        for k in range(dec.shape[0]):
            ssim_sum += _ssim((dec[k:k + 1] + 1) / 2, (gt[k:k + 1] + 1) / 2,
                              1.0, 11, win)
        cnt += (j - i)
    del win
    return mse_sum / cnt, ssim_sum / cnt


def eval_ckpt(ctrl, vae, diffusion, device, cache, ckpt_dir, step, cfg_params):
    """Evaluate one ckpt: base (no skel) + ctrl (with skel)."""
    n = len(cache["conds"])
    t0 = time.time()

    # 1) base: 无 skel (退化为主模型)
    log(f"[eval] step {step}: base (no skel) ...")
    mse_base, ssim_base = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        steps=cfg_params["steps"], cfg=cfg_params["cfg"],
        seed=cfg_params["seed"], batch=cfg_params["batch"], use_skel=False)
    log(f"[eval] step {step}: base  MSE={mse_base:.5f} SSIM={ssim_base:.4f}")

    # 2) ctrl: 有 GT skel
    log(f"[eval] step {step}: ctrl (GT skel) ...")
    mse_ctrl, ssim_ctrl = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        steps=cfg_params["steps"], cfg=cfg_params["cfg"],
        seed=cfg_params["seed"], batch=cfg_params["batch"], use_skel=True)
    log(f"[eval] step {step}: ctrl  MSE={mse_ctrl:.5f} SSIM={ssim_ctrl:.4f}")

    result = {
        "step": step,
        "mse_base": mse_base, "ssim_base": ssim_base,
        "mse_ctrl": mse_ctrl, "ssim_ctrl": ssim_ctrl,
        "delta_mse": mse_ctrl - mse_base,
        "delta_ssim": ssim_ctrl - ssim_base,
    }
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[eval] step {step}: done ({time.time()-t0:.0f}s) "
        f"ΔMSE={result['delta_mse']:+.5f} ΔSSIM={result['delta_ssim']:+.4f}")
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
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"[init] device={device}, eval_csv={args.eval_csv}, n={args.eval_n}")

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # 加载共享组件 (只一次): 主模型 + ctrl shell + VAE + cache + diffusion
    log("[init] building main model (195k) ...")
    main_model = build_main_model(device)
    log("[init] building ctrl shell ...")
    ctrl = build_ctrl(main_model, device)
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
