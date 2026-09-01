# -*- coding: utf-8 -*-
"""
CPU metrics daemon for ControlNet evals: watches for eval_pending_ctrl_*.json
markers, reads pred/gt PNG pairs from eval_samples_ctrl/stepXXXXXXX/{base,ctrl}/,
computes MSE/SSIM/skel_iou/LPIPS (mean/std/min + quantiles), writes
eval_auto_ctrl_{step}.json with base/ctrl/delta fields.

Reuses the metric conventions of eval_metrics_daemon.py.
"""
import os, json, glob, time, sys, traceback
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ctrl-metrics] {msg}", flush=True)


_lpips_fn = None
_lpips_loaded = False

def _get_lpips():
    global _lpips_fn, _lpips_loaded
    if _lpips_loaded:
        return _lpips_fn
    _lpips_loaded = True
    try:
        import lpips
        import torch
        _lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        _lpips_fn.eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
        _log("LPIPS model loaded (vgg, CPU)")
    except Exception as e:
        _log(f"LPIPS unavailable ({e}), will skip lpips metric")
        _lpips_fn = None
    return _lpips_fn


def _mse(pred, gt):
    return float(np.mean((pred - gt) ** 2)) * 4.0


def _ssim(pred, gt, win=11, data_range=1.0):
    from scipy.ndimage import correlate1d
    radius = win // 2
    x_k = np.arange(-radius, radius + 1, dtype=np.float64)
    k1d = np.exp(-(x_k ** 2) / (2 * 1.5 ** 2))
    k1d /= k1d.sum()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssims = []
    for ch in range(pred.shape[2]):
        x = pred[:, :, ch].astype(np.float64)
        y = gt[:, :, ch].astype(np.float64)
        def _g(img):
            return correlate1d(correlate1d(img, k1d, axis=0, mode='reflect'),
                               k1d, axis=1, mode='reflect')
        mu_x = _g(x); mu_y = _g(y)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sx2 = _g(x * x) - mu_x2
        sy2 = _g(y * y) - mu_y2
        sxy = _g(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def _skel_iou(pred, gt, thresh=0.5):
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
    g1 = pred.mean(axis=2); g2 = gt.mean(axis=2)
    b1 = g1 < thresh; b2 = g2 < thresh
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    s1 = skeletonize(b1); s2 = skeletonize(b2)
    inter = float((s1 & s2).sum())
    union = float((s1 | s2).sum())
    return inter / union if union > 0 else 1.0


def _lpips_score(pred_np, gt_np, lpips_fn):
    if lpips_fn is None:
        return None
    import torch
    p = torch.from_numpy(pred_np.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    g = torch.from_numpy(gt_np.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        return float(lpips_fn(p, g).item())


def compute_metrics_for_dir(sample_dir, n, prefix="sample"):
    """Compute metrics for prefix{i}.png vs gt{i}.png in sample_dir."""
    lpips_fn = _get_lpips()
    mses, ssims, skels, lpipses = [], [], [], []
    for i in range(n):
        pred_path = os.path.join(sample_dir, f"{prefix}{i}.png")
        gt_path = os.path.join(sample_dir, f"gt{i}.png")
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        pred = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        mses.append(_mse(pred, gt))
        ssims.append(_ssim(pred, gt))
        skels.append(_skel_iou(pred, gt))
        lp = _lpips_score(pred, gt, lpips_fn)
        if lp is not None:
            lpipses.append(lp)
    if not mses:
        return None
    def _q(arr, p):
        a = sorted(arr)
        if not a:
            return None
        k = (len(a) - 1) * p
        f = int(np.floor(k)); c = int(np.ceil(k))
        if f == c:
            return float(a[f])
        return float(a[f] * (c - k) + a[c] * (k - f))
    result = {
        "mse": float(np.mean(mses)),
        "ssim": float(np.mean(ssims)),
        "skel_iou": float(np.mean(skels)),
        "mse_std": float(np.std(mses)),
        "ssim_std": float(np.std(ssims)),
        "skel_iou_std": float(np.std(skels)),
        "ssim_min": float(np.min(ssims)),
        "skel_iou_min": float(np.min(skels)),
        "mse_q25": _q(mses, 0.25), "mse_q50": _q(mses, 0.50), "mse_q75": _q(mses, 0.75),
        "ssim_q25": _q(ssims, 0.25), "ssim_q50": _q(ssims, 0.50), "ssim_q75": _q(ssims, 0.75),
        "skel_iou_q25": _q(skels, 0.25), "skel_iou_q50": _q(skels, 0.50), "skel_iou_q75": _q(skels, 0.75),
        "n": len(mses),
    }
    if lpipses:
        result["lpips"] = float(np.mean(lpipses))
        result["lpips_std"] = float(np.std(lpipses))
        result["lpips_min"] = float(np.min(lpipses))
        result["lpips_q25"] = _q(lpipses, 0.25)
        result["lpips_q50"] = _q(lpipses, 0.50)
        result["lpips_q75"] = _q(lpipses, 0.75)
    return result


def process_pending(pending_path, ckpt_dir):
    with open(pending_path) as f:
        info = json.load(f)
    step = info["step"]
    step_tag = info["step_tag"]
    n = info["n"]
    base_dir = os.path.join(ckpt_dir, "eval_samples_ctrl", step_tag, "base")
    ctrl_dir = os.path.join(ckpt_dir, "eval_samples_ctrl", step_tag, "ctrl")
    _log(f"step {step}: computing base/ctrl metrics (n={n})")
    t0 = time.time()
    res_base = compute_metrics_for_dir(base_dir, n, prefix="base")
    res_ctrl = compute_metrics_for_dir(ctrl_dir, n, prefix="ctrl")
    if res_base is None or res_ctrl is None:
        _log(f"step {step}: missing images, skipping")
        return
    result = {
        "step": step,
        "base": res_base,
        "ctrl": res_ctrl,
        "delta_mse": res_ctrl["mse"] - res_base["mse"],
        "delta_ssim": res_ctrl["ssim"] - res_base["ssim"],
        "delta_skel_iou": res_ctrl["skel_iou"] - res_base["skel_iou"],
        "n": n,
    }
    if "lpips" in res_base and "lpips" in res_ctrl:
        result["delta_lpips"] = res_ctrl["lpips"] - res_base["lpips"]
    out_path = os.path.join(ckpt_dir, f"eval_auto_ctrl_{int(step):07d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    _log(f"step {step}: base  MSE={res_base['mse']:.5f} SSIM={res_base['ssim']:.4f} "
         f"SkelIoU={res_base['skel_iou']:.4f}")
    _log(f"step {step}: ctrl  MSE={res_ctrl['mse']:.5f} SSIM={res_ctrl['ssim']:.4f} "
         f"SkelIoU={res_ctrl['skel_iou']:.4f}")
    _log(f"step {step}: ΔMSE={result['delta_mse']:+.5f} ΔSSIM={result['delta_ssim']:+.4f} "
         f"ΔSkelIoU={result['delta_skel_iou']:+.4f} ({time.time()-t0:.1f}s)")
    # cleanup marker
    os.remove(pending_path)


def _find_all_ckpt_dirs(results_dir):
    dirs = []
    for sub in sorted(os.listdir(results_dir)):
        ck = os.path.join(results_dir, sub, "checkpoints")
        if os.path.isdir(ck):
            dirs.append(ck)
    return dirs


def main(rd):
    _log(f"watching {rd} for eval_pending_ctrl_*.json markers")
    os.makedirs(rd, exist_ok=True)
    while True:
        try:
            for ckpt_dir in _find_all_ckpt_dirs(rd):
                pending = sorted(glob.glob(os.path.join(ckpt_dir, "eval_pending_ctrl_*.json")))
                for p in pending:
                    try:
                        process_pending(p, ckpt_dir)
                    except Exception as e:
                        _log(f"ERROR processing {p}: {e}")
                        _log(traceback.format_exc())
        except Exception as e:
            _log(f"ERROR in watch loop: {e} (will retry)")
            _log(traceback.format_exc())
        time.sleep(20)


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "5script/results/ctrl_skel"
    main(rd)