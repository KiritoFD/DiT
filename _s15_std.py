import json, glob, sys
sys.stdout.reconfigure(encoding="utf-8")
fs = sorted(glob.glob("5script/results/s15_ws_flow/*/checkpoints/eval_auto_*.json"))
# print keys of one file
d0 = json.load(open(fs[len(fs)//2]))
print("keys:", sorted(d0.keys()))
print()
# focus on mse_std, ssim_std, ssim_min, skel_iou_min, skel_iou_std across time
print("step\tmse\tmse_std\tssim\tssim_std\tssim_min\tskel\tskel_std\tskel_min\tn")
for f in fs:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d.get("step")
    print(f"{s}\t{d.get('mse')}\t{d.get('mse_std')}\t{d.get('ssim')}\t{d.get('ssim_std')}\t{d.get('ssim_min')}\t"
          f"{d.get('skel_iou')}\t{d.get('skel_iou_std')}\t{d.get('skel_iou_min')}\t{d.get('n')}")
