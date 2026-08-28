import json, glob, sys
sys.stdout.reconfigure(encoding="utf-8")
fs = sorted(glob.glob("5script/results/s17_s_flow/*/checkpoints/eval_auto_*.json"))
rows = []
for f in fs:
    d = json.load(open(f))
    rows.append({
        "step": d.get("step"), "mse": d.get("mse"), "ssim": d.get("ssim"),
        "mse_std": d.get("mse_std"), "ssim_min": d.get("ssim_min"),
        "ssim_std": d.get("ssim_std"), "skel_iou": d.get("skel_iou"),
        "skel_iou_min": d.get("skel_iou_min"), "lpips": d.get("lpips")})
rows.sort(key=lambda r: r["step"])
print("step\tmse\tmse_std\tssim\tssim_min\tssim_std\tskel\tskel_min\tlpips")
for r in rows:
    print(f"{r['step']}\t{round(r['mse'],4)}\t{round(r['mse_std'],4)}\t{round(r['ssim'],4)}\t"
          f"{r['ssim_min']:.4f}\t{round(r['ssim_std'],4)}\t{round(r['skel_iou'],4)}\t"
          f"{r['skel_iou_min']:.4f}\t{round(r['lpips'],4)}")
# Best by ssim and by combo selection criteria
best_ssim = max(rows, key=lambda r: r["ssim"] or 0)
print("\nBEST ssim:", best_ssim["step"], "->", round(best_ssim["ssim"],4))
best_skel = max(rows, key=lambda r: r["skel_iou"] or 0)
print("BEST skel:", best_skel["step"], "->", round(best_skel["skel_iou"],4))
best_tail = max(rows, key=lambda r: (r["skel_iou"] or 0) - (r["ssim_min"] or 0))
print("BEST combo(tail-weighted):", best_tail["step"])
# Also: latest eval_auto has ssim_min which is most negative -> tail degradation
for r in rows[-3:]:
    print("tail:", r["step"], "ssim_min:", round(r["ssim_min"],4), "ssim:", round(r["ssim"],4))