"""CFG sweep on best ctrl ckpt. Load model once, eval 4 cfg values."""
import os, sys, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

import torch
import torch.nn.functional as F

# ---- import from auto_eval_ctrl ----
from auto_eval_ctrl import (
    build_main_model, build_ctrl, load_vae, build_cache,
    load_ctrl_weights, eval_one_step, _get_lpips, log
)
from diffusion import create_diffusion

# ---- args ----
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="ctrl ckpt path")
ap.add_argument("--cfgs", default="1.0,1.7,2.5,4.0")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-csv", default="5script/eval100_top6.csv")
ap.add_argument("--eval-n", type=int, default=100)
ap.add_argument("--out", default=None, help="output json path")
args = ap.parse_args()

cfgs = [float(x) for x in args.cfgs.split(",")]
device = torch.device("cuda")

log(f"[init] device={device}, ckpt={args.ckpt}")
log(f"[init] cfg sweep: {cfgs}")

# ---- load components once ----
log("[init] building main model ...")
main_model = build_main_model(device, from_scratch=False)
log("[init] building ctrl shell ...")
ctrl = build_ctrl(main_model, device, train_ctrl_only=True)
log("[init] loading VAE ...")
vae = load_vae(device)
log("[init] building eval cache ...")
cache = build_cache(args.eval_csv, n=args.eval_n)
cache["gts"] = cache["gts"].to(device)
cache["skels"] = cache["skels"].to(device)
diffusion = create_diffusion(str(args.steps))
log("[init] loading LPIPS ...")
_get_lpips()

# ---- load ctrl weights ----
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
step = int(ckpt.get("train_steps", 0) or 0)
load_ctrl_weights(ctrl, ckpt, os.path.basename(args.ckpt))
log(f"[load] step {step} loaded")

# ---- sweep ----
results = {"ckpt": args.ckpt, "step": step, "cfgs": cfgs, "sweep": {}}
for cfg in cfgs:
    log(f"[sweep] cfg={cfg} ...")
    t0 = time.time()
    # base (no skel)
    mse_b, ssim_b, lpips_b, skel_b = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=args.eval_n,
        steps=args.steps, cfg=cfg, seed=args.seed, batch=args.batch, use_skel=False)
    # ctrl (GT skel)
    mse_c, ssim_c, lpips_c, skel_c = eval_one_step(
        ctrl, vae, diffusion, device, cache, n=args.eval_n,
        steps=args.steps, cfg=cfg, seed=args.seed, batch=args.batch, use_skel=True)
    r = {
        "cfg": cfg,
        "mse_base": mse_b, "ssim_base": ssim_b, "lpips_base": lpips_b, "skel_iou_base": skel_b,
        "mse_ctrl": mse_c, "ssim_ctrl": ssim_c, "lpips_ctrl": lpips_c, "skel_iou_ctrl": skel_c,
        "delta_ssim": ssim_c - ssim_b, "delta_mse": mse_c - mse_b,
        "delta_lpips": (lpips_c - lpips_b) if lpips_b is not None and lpips_c is not None else None,
        "delta_skel_iou": skel_c - skel_b,
    }
    results["sweep"][str(cfg)] = r
    log(f"[sweep] cfg={cfg}: base_ssim={ssim_b:.4f} ctrl_ssim={ssim_c:.4f} d={ssim_c-ssim_b:+.4f} "
        f"base_lpips={lpips_b:.4f} ctrl_lpips={lpips_c:.4f} d_lpips={lpips_c-lpips_b:+.4f} "
        f"base_skel={skel_b:.4f} ctrl_skel={skel_c:.4f} d_skel={skel_c-skel_b:+.4f} [{time.time()-t0:.0f}s]")
    print(f"cfg={cfg}: base_ssim={ssim_b:.4f} ctrl_ssim={ssim_c:.4f} d_ssim={ssim_c-ssim_b:+.4f} "
          f"base_lpips={lpips_b:.4f} ctrl_lpips={lpips_c:.4f} base_skel={skel_b:.4f} ctrl_skel={skel_c:.4f}", flush=True)

# ---- save ----
out_path = args.out or os.path.join(os.path.dirname(args.ckpt), f"cfg_sweep_{step:07d}.json")
with open(out_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
log(f"[done] saved {out_path}")
print(f"\n=== CFG SWEEP SUMMARY (step {step}) ===", flush=True)
print(f"{'cfg':>6} {'base_ssim':>10} {'ctrl_ssim':>10} {'Δssim':>8} {'base_lpips':>10} {'ctrl_lpips':>10} {'Δlpips':>8} {'base_skel':>9} {'ctrl_skel':>9} {'Δskel':>8}")
for cfg in cfgs:
    r = results["sweep"][str(cfg)]
    print(f"{cfg:6.1f} {r['ssim_base']:10.4f} {r['ssim_ctrl']:10.4f} {r['delta_ssim']:8.4f} "
          f"{r['lpips_base']:10.4f} {r['lpips_ctrl']:10.4f} {r['delta_lpips']:8.4f} "
          f"{r['skel_iou_base']:9.4f} {r['skel_iou_ctrl']:9.4f} {r['delta_skel_iou']:8.4f}")
