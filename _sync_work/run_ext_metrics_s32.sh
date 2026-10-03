#!/bin/bash
# run_ext_metrics_s32.sh — 对 s32 REPA 各 eval step 的 ctrl/base 图跑扩展指标, 汇总对比
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
S32=$(ls -dt assets/results/s32_repa_finetune/*/ 2>/dev/null | head -1)
echo "S32=$S32"

# 1) 逐个 step 计算扩展指标 (写每个目录的 metrics_ext.json)
for d in $(ls -d "$S32/checkpoints/eval_samples_ctrl"/step* 2>/dev/null | sort); do
  STEP=$(basename "$d" | tr -d 'step')
  for tag in ctrl base; do
    if [ -d "$d/$tag" ] && [ -n "$(ls "$d/$tag"/*.png 2>/dev/null)" ]; then
      N=$(ls "$d/$tag"/${tag}*.png 2>/dev/null | wc -l)
      [ "$N" -eq 0 ] && continue
      if [ -f "$d/$tag/metrics_ext.json" ]; then
        echo "[skip] s32 step $STEP $tag (已有)"
        continue
      fi
      echo "[metrics] s32 step $STEP $tag (n=$N) ..."
      $PY src/eval/metrics_png.py --dir "$d/$tag" --tag "$tag" --n "$N" --out "$d/$tag/metrics_ext.json" \
         >> /tmp/ext_metrics_s32.log 2>&1 || { echo "  FAILED step=$STEP tag=$tag"; tail -5 /tmp/ext_metrics_s32.log; continue; }
    fi
  done
done

# 2) 汇总 (统一由 python 收集, 不拼 shell 字符串)
$PY - <<'EOF'
import json, glob, os
S32 = "/root/Workspace/xy/DiT/assets/results/s32_repa_finetune/20260901-183011-s32-repa-finetune"
rows = []
for m in sorted(glob.glob(os.path.join(S32, "checkpoints/eval_samples_ctrl/step*/[bc]*/metrics_ext.json"))):
    step = int(os.path.basename(os.path.dirname(os.path.dirname(m))).replace("step", ""))
    tag = os.path.basename(os.path.dirname(m))
    j = json.load(open(m))
    rows.append({"step": step, "tag": tag,
        "mse": j["mse"]["mean"], "psnr": j["psnr"]["mean"], "ssim": j["ssim"]["mean"],
        "tv": j["tv"]["mean"], "lap_var": j["lap_var"]["mean"],
        "hf_energy": j["hf_energy"]["mean"], "saltpepper": j["saltpepper"]["mean"],
        "edge_clean": j["edge_clean"]["mean"],
        "lpips": j.get("lpips", {}).get("mean", 0.0)})
rows.sort(key=lambda r: (r["step"], r["tag"]))
out = os.path.join(S32, "checkpoints/ext_metrics_summary.json")
json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)
if not rows:
    print("(no metrics rows)")
    raise SystemExit
hdr = f"{'step':>6} {'tag':>5} | {'mse':>8} {'psnr':>6} {'ssim':>7} {'tv':>8} {'lap':>8} {'hf':>8} {'s&p':>8} {'edge':>7} {'lpips':>7}"
print(hdr)
for r in rows:
    print(f"{r['step']:>6} {r['tag']:>5} | {r['mse']:>8.4f} {r['psnr']:>6.2f} {r['ssim']:>7.4f} {r['tv']:>8.5f} {r['lap_var']:>8.5f} {r['hf_energy']:>8.5f} {r['saltpepper']:>8.5f} {r['edge_clean']:>7.4f} {r['lpips']:>7.4f}")
print(f"\n-> {out}")
EOF