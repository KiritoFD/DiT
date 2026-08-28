import json, glob, os
fs = sorted(glob.glob("5script/results/s17_s_flow/*/checkpoints/eval_auto_*.json"))
rows = []
for f in fs:
    d = json.load(open(f))
    rows.append((d.get("step"), d.get("mse"), d.get("ssim"), d.get("skel_iou"), d.get("lpips")))
rows.sort()
print("step\tmse\tssim\tskel_iou\tlpips")
for r in rows:
    print(f"{r[0]}\t{round(r[1],4) if r[1] else '-'}\t{round(r[2],4) if r[2] else '-'}\t{round(r[3],4) if r[3] else '-'}\t{round(r[4],4) if r[4] else '-'}")
# best ssim and where
best = max(rows, key=lambda r: r[2] or 0)
print("BEST ssim:", best)
