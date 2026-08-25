import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
with open("train_data.json", encoding="utf-8") as f:
    data = json.load(f)
rows = data["rows"]
eval_rows = [r for r in rows if r.get("is_eval")]
print(f"total rows: {len(rows)}, eval rows: {len(eval_rows)}")
for r in eval_rows:
    lp = r.get("lpips")
    lp_s = f"{lp:.4f}" if lp is not None else "---"
    mse = r.get("mse", 0)
    ssim = r.get("ssim", 0)
    si = r.get("skel_iou", 0)
    print(f"  step={r['step']:6d} mse={mse:.5f} ssim={ssim:.4f} skel_iou={si:.4f} lpips={lp_s}")
