#!/bin/bash
# run_ext_metrics_s32b.sh — 对 s32b REPA 各 eval step 跑扩展视觉质量指标, 与 REPA 前基线(s31@42500)对比
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
S32B=$(ls -dt assets/results/s32b_repa_strong/*/ 2>/dev/null | head -1)
BASE=assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/manual_eval_correct_skel/step0042500/ctrl

echo "S32B=$S32B"
echo "=== 1) REPA 前基线 (s31@42500, cfg=0.7) ==="
if [ -f "$BASE/metrics_ext.json" ]; then
  echo "[已有] s31@42500 metrics_ext.json"
else
  $PY src/eval/metrics_png.py --dir "$BASE" --tag ctrl --n 100 --out "$BASE/metrics_ext.json" > /tmp/ext_base_s31.log 2>&1
  echo "[ok] 计算完成"
fi

echo "=== 2) s32b 各 eval step ==="
for d in $(ls -d "$S32B/checkpoints/eval_samples_ctrl"/step* 2>/dev/null | sort); do
  STEP=$(basename "$d" | tr -d 'step')
  for tag in ctrl; do
    DD="$d/$tag"
    [ -d "$DD" ] || continue
    N=$(ls "$DD"/${tag}*.png 2>/dev/null | wc -l)
    [ "$N" -eq 0 ] && continue
    if [ -f "$DD/metrics_ext.json" ]; then
      echo "[skip] s32b step $STEP $tag (已有)"
      continue
    fi
    echo "[metrics] s32b step $STEP $tag (n=$N) ..."
    $PY src/eval/metrics_png.py --dir "$DD" --tag "$tag" --n "$N" --out "$DD/metrics_ext.json" >> /tmp/ext_metrics_s32b.log 2>&1 || echo "  FAILED step=$STEP"
  done
done

echo
echo "=== 3) 汇总对比 ==="
$PY - <<'EOF'
import json, os, glob
BASE = "/root/Workspace/xy/DiT/assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/manual_eval_correct_skel/step0042500/ctrl/metrics_ext.json"
S32B = "/root/Workspace/xy/DiT/assets/results/s32b_repa_strong/20260901-204250-s32b-repa-strong"

def pick(j):
    return {k: j[k]["mean"] for k in ["mse","psnr","ssim","tv","lap_var","hf_energy","saltpepper","edge_clean","bg_uniformity","ink_purity","ringing"] if k in j}

rows = []
if os.path.exists(BASE):
    rows.append(("REPA前(s31@42500)", pick(json.load(open(BASE)))))
for m in sorted(glob.glob(os.path.join(S32B, "checkpoints/eval_samples_ctrl/step*/ctrl/metrics_ext.json"))):
    step = os.path.basename(os.path.dirname(os.path.dirname(m))).replace("step", "")
    rows.append((f"s32b@{step}", pick(json.load(open(m)))))

if not rows:
    print("(无指标数据)"); raise SystemExit

keys = ["mse","psnr","ssim","tv","lap_var","hf_energy","saltpepper","edge_clean","bg_uniformity","ink_purity","ringing"]
hdr = f"{'阶段':>16} |" + "".join(f"{k:>12}" for k in keys)
print(hdr)
print("-"*len(hdr))
for name, r in rows:
    line = f"{name:>16} |"
    for k in keys:
        v = r.get(k, float('nan'))
        if k in ("psnr",): line += f"{v:>12.2f}"
        elif k in ("mse","tv","lap_var","hf_energy","saltpepper"): line += f"{v:>12.5f}"
        else: line += f"{v:>12.4f}"
    print(line)

# 训练前(基线) vs 最新 s32b 的 delta
if len(rows) >= 2:
    base_r = rows[0][1]; latest_r = rows[-1][1]
    print("\n=== 训练后 vs 训练前 delta ===")
    print(f"{'指标':>16} {'训练前':>12} {'最新':>12} {'Δ':>12}  {'方向'}")
    dir_map = {"mse": "↓好", "tv": "↓好", "hf_energy": "↓好", "saltpepper": "↓好", "lap_var": "↓好" if latest_r.get("lap_var",0)<=base_r.get("lap_var",0) else "~", "edge_clean": "↓好", "bg_uniformity": "↓好", "ink_purity": "↑好", "ringing": "↑好", "psnr": "↑好", "ssim": "↑好"}
    for k in keys:
        b = base_r.get(k, float('nan')); l = latest_r.get(k, float('nan'))
        d = l - b
        # 方向注释
        good = {"mse": d<0, "tv": d<0, "hf_energy": d<0, "saltpepper": d<0, "edge_clean": d<0, "bg_uniformity": d<0, "lap_var": d<0, "ink_purity": d>0, "ringing": d>0, "psnr": d>0, "ssim": d>0}
        arrow = "✓改善" if good.get(k, False) else ("✗退化" if abs(d) > 1e-4 else "≈持平")
        print(f"{k:>16} {b:>12.4f} {l:>12.4f} {d:>+12.4f}  {arrow}")
EOF