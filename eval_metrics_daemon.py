"""
CPU-only eval metrics daemon: watches for eval_pending_*.json markers,
reads pred+gt PNGs from disk, computes MSE/SSIM/skel_iou/LPIPS, writes eval_auto_*.json.

Run as a separate process (or background thread) alongside training.
CPU-only — does NOT touch GPU at all.

LPIPS uses the lpips package (torch VGG/AlexNet backbone, runs on CPU).
"""
import os, json, glob, time, sys, traceback
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))


def _log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [eval-metrics] {msg}", flush=True)


# ── LPIPS model (lazy singleton, CPU) ──────────────────────────────────────────

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


# ── Metrics (numpy, CPU) ──────────────────────────────────────────────────────

def _mse(pred, gt):
    """pred, gt: (H,W,3) float32 [0,1]. Returns MSE in [-1,1] range (×4).
    This matches the old eval_gen.py convention: F.mse_loss on [-1,1] tensors.
    [0,1] MSE × 4 = [-1,1] MSE because (2a-2b)^2 = 4(a-b)^2."""
    return float(np.mean((pred - gt) ** 2)) * 4.0


def _ssim(pred, gt, win=11, data_range=1.0):
    """SSIM for (H,W,3) arrays. Gaussian window (matches eval_auto.py).

    Uses separable Gaussian filtering for speed; equivalent to the 2D Gaussian
    conv2d in eval_auto.py (_gaussian_window σ=1.5, size=11).
    """
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
        mu_x = _g(x)
        mu_y = _g(y)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sx2 = _g(x * x) - mu_x2
        sy2 = _g(y * y) - mu_y2
        sxy = _g(x * y) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
                   ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
        ssims.append(ssim_map.mean())
    return float(np.mean(ssims))


def _skel_iou(pred, gt, thresh=0.5):
    """Skeleton IoU: binarize, skeletonize, IoU of skeletons."""
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

    g1 = pred.mean(axis=2)
    g2 = gt.mean(axis=2)
    b1 = g1 < thresh
    b2 = g2 < thresh
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    s1 = skeletonize(b1)
    s2 = skeletonize(b2)
    inter = float((s1 & s2).sum())
    union = float((s1 | s2).sum())
    return inter / union if union > 0 else 1.0


def _lpips_score(pred_np, gt_np, lpips_fn):
    """Compute LPIPS for a single pair. pred_np/gt_np: (H,W,3) float32 [0,1].
    LPIPS expects [-1,1] (N,3,H,W) tensors."""
    if lpips_fn is None:
        return None
    import torch
    # (H,W,3) -> (1,3,H,W), scale [0,1] -> [-1,1]
    p = torch.from_numpy(pred_np.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    g = torch.from_numpy(gt_np.transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
    with torch.no_grad():
        val = float(lpips_fn(p, g).item())
    return val


def compute_metrics_for_dir(sample_dir, n):
    """Read n pairs of sample{i}.png + gt{i}.png, compute metrics."""
    lpips_fn = _get_lpips()
    mses, ssims, skels, lpipses = [], [], [], []
    for i in range(n):
        pred_path = os.path.join(sample_dir, f"sample{i}.png")
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

    result = {
        "mse": float(np.mean(mses)),
        "ssim": float(np.mean(ssims)),
        "skel_iou": float(np.mean(skels)),
        "mse_std": float(np.std(mses)),
        "ssim_std": float(np.std(ssims)),
        "skel_iou_std": float(np.std(skels)),
        "ssim_min": float(np.min(ssims)),
        "skel_iou_min": float(np.min(skels)),
        "n": len(mses),
    }
    if lpipses:
        result["lpips"] = float(np.mean(lpipses))
        result["lpips_std"] = float(np.std(lpipses))
        result["lpips_min"] = float(np.min(lpipses))
    return result


def process_one(pending_path, checkpoint_dir):
    """Process one eval_pending_*.json → compute metrics → eval_auto_*.json."""
    with open(pending_path) as f:
        info = json.load(f)
    step = info["step"]
    n = info["n"]
    sample_dir = os.path.join(checkpoint_dir, info["dir"])

    _log(f"step {step}: computing metrics for {n} images in {sample_dir}")
    t0 = time.time()

    result = compute_metrics_for_dir(sample_dir, n)
    if result is None:
        _log(f"step {step}: no images found, skipping")
        return

    result["step"] = step
    result["elapsed_metrics"] = time.time() - t0
    result["elapsed_gpu"] = info.get("elapsed_gpu", 0)

    out_path = os.path.join(checkpoint_dir, f"eval_auto_{int(step):07d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    _lpips_str = f" LPIPS={result['lpips']:.4f}" if "lpips" in result else ""
    _log(f"step {step}: MSE={result['mse']:.5f} SSIM={result['ssim']:.4f} "
         f"SkelIoU={result['skel_iou']:.4f}{_lpips_str} ({result['elapsed_metrics']:.1f}s)")

    # Remove pending marker
    os.remove(pending_path)


def _find_all_ckpt_dirs(results_dir):
    """Find ALL ckpt dirs under the series dir, not just the active one.
    This lets the daemon process pending markers from prior runs too."""
    import re
    dirs = []
    series = os.path.join(BASE, results_dir) if not os.path.isabs(results_dir) else results_dir
    if not os.path.isdir(series):
        return dirs
    for run_name in sorted(os.listdir(series)):
        run_dir = os.path.join(series, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            dirs.append(ckpt_dir)
    return dirs


def main(results_dir=None, poll_interval=10):
    """Watch for eval_pending_*.json across ALL ckpt dirs in the series."""
    if results_dir is None:
        results_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not results_dir:
        _log("Usage: eval_metrics_daemon.py <results_dir>")
        return

    _log(f"watching {results_dir} for eval pending markers (scans all run dirs)")

    while True:
        # Scan ALL ckpt dirs in the series (active + prior runs)
        all_ckpt_dirs = _find_all_ckpt_dirs(results_dir)
        for ckpt_dir in all_ckpt_dirs:
            pending = sorted(glob.glob(os.path.join(ckpt_dir, "eval_pending_*.json")))
            for p in pending:
                try:
                    process_one(p, ckpt_dir)
                except Exception as e:
                    _log(f"ERROR processing {p}: {e}")
                    _log(traceback.format_exc())

        time.sleep(poll_interval)


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "5script/results/s10_b4_grey_clear"
    main(rd)
