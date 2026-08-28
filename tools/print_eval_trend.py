# -*- coding: utf-8 -*-
"""Print s19 eval metric trend from eval_auto_*.json files."""
import json, glob, os, sys

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(ckpt_dir, "eval_auto_*.json")))
print(f"{'step':>8} {'MSE':>8} {'SSIM':>8} {'SkelIoU':>8} {'LPIPS':>8} {'n':>5}")
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    step = os.path.basename(f).replace("eval_auto_", "").replace(".json", "")
    print(f"{step:>8} {d.get('mse', 0):8.4f} {d.get('ssim', 0):8.4f} "
          f"{d.get('skel_iou', 0):8.4f} {d.get('lpips', 0):8.4f} {d.get('n', 0):>5}")
