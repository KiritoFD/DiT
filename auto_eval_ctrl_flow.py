# -*- coding: utf-8 -*-
"""
(auto) eval for flow ControlNet — free sampling with flow Euler + VAE decode,
mirrors auto_eval_ctrl.eval_one_step but flow-aware. GPU/CPU device arg.

Metrics: MSE / SSIM / LPIPS / skeleton-IoU on GT, base (no skel) vs ctrl (skel).
Writes per-step JSON into the active ckpt dir.
"""
import os, sys, argparse, json, time, datetime, glob
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

SKEL_ROOT = "final_skeleton_d3"
IMG_ROOT = "final_imgs_256"
DEFAULT_MAIN_CKPT = "5script/results/s6_top6_diffonly/20260820-191536-s6-top6-diffonly-resume/checkpoints/0195000.pt"
DEFAULT_EVAL_CSV = "5script/eval100_top6.csv"
DEFAULT_VAE = "pretrained_models/sd-vae-ft-ema"


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _gaussian_kernel_1d(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _ssim_batch(pred, gt, win=11, data_range=1.0, sigma=1.5):
    from scipy.ndimage import correlate1d
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    radius = win // 2
    k1d = _gaussian_kernel_1d(sigma, radius)
    n = pred.shape[0]
    x = pred.reshape(-1, pred.shape[2], pred.shape[3]).astype(np.float64)
    y = gt.reshape(-1, gt.shape[2], gt.shape[3]).astype(np.float64)
    def _gauss_blur(img):
        return correlate1d(correlate1d(img, k1d, axis=1, mode='reflect'),
                           k1d, axis=2, mode='reflect')
    mu_x = _gauss_blur(x); mu_y = _gauss_blur(y)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = _gauss_blur(x * x) - mu_x2
    sy2 = _gauss_blur(y * y) - mu_y2
    sxy = _gauss_blur(x * y) - mu_xy
    m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
    per_img = m.reshape(n, 3, -1).mean(axis=(1, 2))
    return float(per_img.mean())


_lpips_fn = None
def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
            _lpips_fn.eval()
            for p in _lpips_fn.parameters():
                p.requires_grad_(False)
            log("LPIPS loaded (vgg)")
        except Exception as e:
            log(f"LPIPS unavailable: {e}")
            _lpips_fn = False
    return _lpips_fn or None


def _skel_iou_batch(pred_np, gt_np, thresh=0.5):
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        def skeletonize(binary):
            skel = np.zeros_like(binary); img = binary.copy()
            st = generate_binary_structure(2, 2)
            while img.any():
                e = binary_erosion(img, structure=st)
                skel |= img & ~e
                img = e
            return skel
    n = pred_np.shape[0]
    g1 = pred_np.mean(axis=3); g2 = gt_np.mean(axis=3)
    b1 = g1 < thresh; b2 = g2 < thresh
    inter_sum = union_sum = 0.0
    for k in range(n):
        if not b1[k].any() and not b2[k].any():
            inter_sum += 1.0; union_sum += 1.0; continue
        if not b1[k].any() or not b2[k].any():
            union_sum += 1.0; continue
        s1 = skeletonize(b1[k]); s2 = skeletonize(b2[k])
        inter_sum += float((s1 & s2).sum()); union_sum += float((s1 | s2).sum())
    return inter_sum / union_sum if union_sum > 0 else 1.0


@torch.no_grad()
def eval_one_step(model, vae, diffusion, device, cache, n=100,
                  cfg=4.0, seed=0, batch=16, use_skel=False):
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z,
                clip_denoised=False, model_kwargs=mk, device=device)
        dec = vae.decode(samples.float() / 0.18215).sample
        gt = gts[i:j]
        mse_sum += torch.nn.functional.mse_loss(dec, gt).item() * (j - i)
        decs.append(dec.cpu()); gts_all.append(gt.cpu())
        cnt += (j - i)
    dec_cat = torch.cat(decs, 0); gt_cat = torch.cat(gts_all, 0)
    dec01 = ((dec_cat + 1) / 2).clamp(0, 1).numpy()
    gt01 = ((gt_cat + 1) / 2).clamp(0, 1).numpy()
    mse = mse_sum / cnt
    ssim = _ssim_batch(dec01, gt01)
    lpips = _lpips_batch(dec_cat, gt_cat, lpips_fn)
    skel_iou = _skel_iou_batch(dec01, gt01)
    return mse, ssim, lpips, skel_iou


def _lpips_batch(pred_t, gt_t, lpips_fn):
    if lpips_fn is None:
        return None
    p = pred_t.float().cpu(); g = gt_t.float().cpu()
    with torch.no_grad():
        return float(lpips_fn(p, g).mean().item())


def eval_ckpt(ctrl, vae, diffusion, device, cache, ckpt_dir, step, cfg_params):
    n = len(cache["conds"])
    t0 = time.time()
    log(f"[eval] step {step}: base (no skel) ...")
    mse_base, ssim_base, lpips_base, skel_base = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        cfg=cfg_params["cfg"], seed=cfg_params["seed"],
        batch=cfg_params["batch"], use_skel=False)
    log(f"[eval] step {step}: base  MSE={mse_base:.5f} SSIM={ssim_base:.4f}"
        + (f" LPIPS={lpips_base:.4f}" if lpips_base is not None else " LPIPS=n/a"))
    log(f"[eval] step {step}: base  SkelIoU={skel_base:.4f}")
    log(f"[eval] step {step}: ctrl (GT skel) ...")
    mse_ctrl, ssim_ctrl, lpips_ctrl, skel_ctrl = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=n,
        cfg=cfg_params["cfg"], seed=cfg_params["seed"],
        batch=cfg_params["batch"], use_skel=True)
    log(f"[eval] step {step}: ctrl  MSE={mse_ctrl:.5f} SSIM={ssim_ctrl:.4f}"
        + (f" LPIPS={lpips_ctrl:.4f}" if lpips_ctrl is not None else " LPIPS=n/a"))
    log(f"[eval] step {step}: ctrl  SkelIoU={skel_ctrl:.4f}")
    result = {
        "step": step,
        "mse_base": mse_base, "ssim_base": ssim_base, "skel_iou_base": skel_base,
        "mse_ctrl": mse_ctrl, "ssim_ctrl": ssim_ctrl, "skel_iou_ctrl": skel_ctrl,
        "delta_mse": mse_ctrl - mse_base, "delta_ssim": ssim_ctrl - ssim_base,
        "delta_skel_iou": skel_ctrl - skel_base,
    }
    if lpips_base is not None: result["lpips_base"] = lpips_base
    if lpips_ctrl is not None: result["lpips_ctrl"] = lpips_ctrl
    if lpips_base is not None and lpips_ctrl is not None:
        result["delta_lpips"] = lpips_ctrl - lpips_base
    with open(os.path.join(ckpt_dir, f"eval_auto_{step:07d}.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"[eval] step {step}: done ({time.time()-t0:.0f}s) "
        f"ΔMSE={result['delta_mse']:+.5f} ΔSSIM={result['delta_ssim']:+.4f} "
        f"ΔSkelIoU={result['delta_skel_iou']:+.4f}")
    return result


def read_active_ckpt_dir(results_dir):
    p = os.path.join(results_dir, "_active_ckpt_dir.txt")
    if os.path.isfile(p):
        with open(p) as f:
            return f.read().strip()
    return None


def build_cache(eval_csv, n=100):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="5script/results/ctrl_skel")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--main-ckpt", default=None)
    ap.add_argument("--char-embed-dim", type=int, default=256)
    ap.add_argument("--char-proj-mode", default="full")
    ap.add_argument("--freeze-char-table", action="store_true")
    ap.add_argument("--diffusion-type", default="ddpm", choices=["ddpm", "flow"])
    args = ap.parse_args()

    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    _ctrl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "controlnet")
    if _ctrl not in sys.path:
        sys.path.insert(0, _ctrl)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log(f"[init] device={device} eval_csv={args.eval_csv} n={args.eval_n} "
        f"diffusion_type={args.diffusion_type} char_embed_dim={args.char_embed_dim} "
        f"char_proj_mode={args.char_proj_mode}")
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    from controlnet_dit import ControlNetDiT, load_main_model
    ckpt_path = args.main_ckpt or DEFAULT_MAIN_CKPT
    log(f"[init] loading main model {ckpt_path}")
    main_model = load_main_model(
        model_name="DiT-2Cond-S/2", ckpt_path=ckpt_path, device=device,
        num_calligraphers=1011, num_characters=35130,
        condition_fusion="factorized_add", callig_embed_dim=128,
        char_embed_dim=args.char_embed_dim, char_proj_mode=args.char_proj_mode,
        freeze_char_table=args.freeze_char_table,
        cond_drop_all_prob=0.05, cond_drop_one_prob=0.25, use_checkpoint=False)
    main_model.eval()
    ctrl = ControlNetDiT(main_model, cond_in_channels=1, train_ctrl_only=True).to(device)
    ctrl.eval()
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(DEFAULT_VAE).to(device).eval()
    cache = build_cache(args.eval_csv, n=args.eval_n)
    if device.type == "cuda":
        cache["gts"] = cache["gts"].to(device)
        cache["skels"] = cache["skels"].to(device)
    from diffusion import create_diffusion_or_flow
    diffusion = create_diffusion_or_flow(str(args.steps), diffusion_type=args.diffusion_type)
    cfg_params = {"steps": args.steps, "cfg": args.cfg,
                  "seed": args.seed, "batch": args.batch}

    def load_ctrl(ctrl, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("ctrl")
        if not sd:
            log(f"[ctrl] {ckpt_path}: no ema/ctrl keys, zero-init")
            return
        ctrl_keys = {k: v for k, v in sd.items() if k.startswith("ctrl_encoder")}
        ctrl.load_state_dict(ctrl_keys, strict=False)
        log(f"[ctrl] loaded {os.path.basename(ckpt_path)} ({len(ctrl_keys)} keys)")

    last_ckpt_dir = None
    state = {}
    while True:
        ckpt_dir = args.ckpt_dir or read_active_ckpt_dir(results_dir)
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            log(f"[wait] no active ckpt dir ({results_dir})")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue
        if ckpt_dir != last_ckpt_dir:
            log(f"[watch] active ckpt dir: {ckpt_dir}")
            last_ckpt_dir = ckpt_dir
            state = {}
            sp = os.path.join(ckpt_dir, "eval_flow_state.json")
            if os.path.isfile(sp):
                state = json.load(open(sp))
        done_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        for pt in done_files:
            base = os.path.basename(pt)
            if base in state:
                continue
            if not os.path.exists(pt + ".done"):
                log(f"[skip] {base}: .done marker missing")
                continue
            step = int(base.replace(".pt", ""))
            log(f"[eval] === processing {base} (step {step}) ===")
            try:
                load_ctrl(ctrl, pt)
                res = eval_ckpt(ctrl, vae, diffusion, device, cache, ckpt_dir,
                                step, cfg_params)
                state[base] = {"step": step, "ok": True,
                               "mse_base": res["mse_base"], "mse_ctrl": res["mse_ctrl"],
                               "ts": datetime.datetime.now().isoformat()}
            except Exception as e:
                import traceback
                log(f"[error] eval {base} failed: {e}")
                traceback.print_exc()
                state[base] = {"step": step, "error": str(e)}
            with open(os.path.join(ckpt_dir, "eval_flow_state.json"), "w") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())