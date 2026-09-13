import json, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = "/root/Workspace/xy/DiT/5script/results/s10_b4_grey_clear"
for f in sorted(glob.glob(os.path.join(base, "*/checkpoints/eval_auto_*.json"))):
    d = json.load(open(f))
    lp = d.get('lpips', '—')
    lp_s = f"{lp:.4f}" if isinstance(lp, float) else str(lp)
    print(f"step={d['step']:6d} MSE={d['mse']:.5f} SSIM={d['ssim']:.4f} SkelIoU={d['skel_iou']:.4f} LPIPS={lp_s}")
print("\nPending markers remaining:")
for f in sorted(glob.glob(os.path.join(base, "*/checkpoints/eval_pending_*.json"))):
    print(f"  {os.path.basename(f)}")
