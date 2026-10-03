#!/bin/bash
# run_ext_metrics_s31.sh — 对 s31 正确 eval 的图跑相同扩展指标 (对比基线)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
S31=$(ls -dt assets/results/s31_ctrl_gt_skel_1px/*/ 2>/dev/null | head -1)
BASE="assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/manual_eval_correct_skel"

for d in $(ls -d "$BASE"/step*/ctrl 2>/dev/null); do
  STEP=$(basename $(dirname "$d") | tr -d 'step')
  if [ -f "$d/metrics_ext.json" ]; then
    echo "[skip] s31 step $STEP ctrl (已有)"
    continue
  fi
  N=$(ls "$d"/ctrl*.png 2>/dev/null | wc -l)
  echo "[metrics] s31 step $STEP ctrl (n=$N) ..."
  $PY src/eval/metrics_png.py --dir "$d" --tag ctrl --n "$N" --out "$d/metrics_ext.json" \
     >> /tmp/ext_metrics_s31.log 2>&1 || echo "  FAILED step=$STEP"
done

$PY - <<'EOF'
import json, glob, os
BASE = "/root/Workspace/xy/DiT/assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/manual_eval_correct_skel"
rows = []
for m in sorted(glob.glob(os.path.join(BASE, "step*/ctrl/metrics_ext.json"))):
    step = int(os.path.basename(os.path.dirname(os.path.dirname(m))).replace("step", ""))
    j = json.load(open(m))
    rows.append({"step": step, "tag": "ctrl",
        "mse": j["mse"]["mean"], "psnr": j["psnr"]["mean"], "ssim": j["ssim"]["mean"],
        "tv": j["tv"]["mean"], "lap_var": j["lap_var"]["mean"],
        "hf_energy": j["hf_energy"]["mean"], "saltpepper": j["saltpepper"]["mean"],
        "edge_clean": j["edge_clean"]["mean"],
        "lpips": j.get("lpips", {}).get("mean", 0.0)})
rows.sort(key=lambda r: r["step"])
out = os.path.join(BASE, "ext_metrics_summary.json")
json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)
if not rows:
    print("(no s31 metrics rows)")
    raise SystemExit
hdr = f"{'step':>6} {'tag':>5} | {'mse':>8} {'psnr':>6} {'ssim':>7} {'tv':>8} {'lap':>8} {'hf':>8} {'s&p':>8} {'edge':>7} {'lpips':>7}"
print(hdr)
for r in rows:
    print(f"{r['step']:>6} {r['tag']:>5} | {r['mse']:>8.4f} {r['psnr']:>6.2f} {r['ssim']:>7.4f} {r['tv']:>8.5f} {r['lap_var']:>8.5f} {r['hf_energy']:>8.5f} {r['saltpepper']:>8.5f} {r['edge_clean']:>7.4f} {r['lpips']:>7.4f}")
print(f"\n-> {out}")
EOF