# -*- coding: utf-8 -*-
import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

# s26 GT skel 1px
print("=== s26 (GT skel 1px) ctrl vs base ===")
files = sorted(glob.glob("assets/results/s26_ctrl_gt_skel/*/checkpoints/eval_auto_ctrl_*.json"))
for f in files:
    d = json.load(open(f))
    step = d.get("step", 0)
    row = {"step": step}
    for br in ("base", "ctrl"):
        if br in d:
            v = d[br]
            row[f"{br}_ssim"] = round(v.get("ssim", 0), 4)
            row[f"{br}_skel_iou"] = round(v.get("skel_iou", 0), 4)
            row[f"{br}_lpips"] = round(v.get("lpips", 0), 4) if v.get("lpips") else None
    print(f"  step {step:>6}: base_ssim={row.get('base_ssim')} ctrl_ssim={row.get('ctrl_ssim')} "
          f"base_skel={row.get('base_skel_iou')} ctrl_skel={row.get('ctrl_skel_iou')}")

print()
print("=== ctrl_fame_1pix_v1 (早期 GT 1px) ===")
files = sorted(glob.glob("assets/results/ctrl_fame_1pix_v1/*/checkpoints/eval_auto_ctrl_*.json"))
for f in files[-6:]:
    d = json.load(open(f))
    step = d.get("step", 0)
    b = d.get("base", {}); c = d.get("ctrl", {})
    print(f"  step {step:>6}: base_ssim={round(b.get('ssim',0),4)} ctrl_ssim={round(c.get('ssim',0),4)} "
          f"base_skel={round(b.get('skel_iou',0),4)} ctrl_skel={round(c.get('skel_iou',0),4)}")

print()
print("=== std skel (标准字形) 结果目录 ===")
for d in ["assets/results/ctrl_fame_std_v1", "assets/results/ctrl_fame_v2", "assets/results/ctrl_skel"]:
    fs = sorted(glob.glob(f"{d}/*/checkpoints/eval_auto_ctrl_*.json"))
    if fs:
        print(f"  {d}: {len(fs)} files, last:")
        for f in fs[-3:]:
            j = json.load(open(f))
            b = j.get("base", {}); c = j.get("ctrl", {})
            print(f"    step {j.get('step',0):>6}: base_ssim={round(b.get('ssim',0),4)} ctrl_ssim={round(c.get('ssim',0),4)} "
                  f"base_skel={round(b.get('skel_iou',0),4)} ctrl_skel={round(c.get('skel_iou',0),4)}")
    else:
        print(f"  {d}: (no eval_auto_ctrl)")
