import json, glob, sys
sys.stdout.reconfigure(encoding="utf-8")
fs = sorted(glob.glob("5script/results/s15_ws_flow/*/checkpoints/eval_auto_*.json"))
rows = []
for f in fs:
    try:
        d = json.load(open(f))
    except Exception as e:
        print("skip", f, e); continue
    rows.append((d.get("step"), d.get("mse"), d.get("ssim"), d.get("skel_iou"), d.get("lpips")))
rows.sort()
print("step\tmse\tssim\tskel_iou\tlpips")
for r in rows:
    print(f"{r[0]}\t{round(r[1],4) if r[1] is not None else '-'}\t{round(r[2],4) if r[2] is not None else '-'}\t{round(r[3],4) if r[3] is not None else '-'}\t{round(r[4],4) if r[4] is not None else '-'}")
# peak ssim
best = max(rows, key=lambda r: r[2] or 0)
print("\nBEST ssim:", best)
print("min mse:", min(rows, key=lambda r: r[1] or 9e9))
print("best skel_iou:", max(rows, key=lambda r: r[3] or 0))
