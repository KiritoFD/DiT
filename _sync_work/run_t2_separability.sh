#!/usr/bin/env bash
# run_t2_separability.sh — T2：生成图的"风格可分性"探针（真实生成图）
#
# 输入：assets/ink_eval/<run>__<step>__strict/（249 张 g{n}.png + 249 张 gt{n}.png）
#       g{n}.png 的数字序 == assets/eval_v13_strict.csv 的行序 -> 可 join 出书家标签
#
# 关键：**留字交叉验证**（group_by=char）+ **同时报 GT 天花板**
#   随机基线 = 1/45 ≈ 0.022。判据：生成 ÷ GT ≥ 0.5 才算"风格真的进了像素"。
#
# 产物：assets/t2_<run>.json + assets/t2_logs/<run>.log
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/eval_v13_strict.csv
mkdir -p assets/t2_logs

for D in assets/ink_eval/*__strict; do
  RUN=$(basename "$D" | sed 's/__strict$//')
  if [ ! -f "$D/g0.png" ]; then echo "[skip] $RUN: 没有 g0.png"; continue; fi
  echo "=================================================================="
  echo "[T2] $RUN"
  echo "=================================================================="
  $PY tools/probe_style_separability.py \
      --from-samples "$D" --eval-csv "$CSV" --group-by char \
      --dino-ckpt data/pretrained/pretrained_models/dinov2_vits14_pretrain.safetensors \
      --out "assets/t2_${RUN}.json" 2>&1 | tee "assets/t2_logs/${RUN}.log"
done

echo
echo "==================== 汇总（生成 vs GT 天花板）===================="
$PY - <<'PYEOF'
import glob, json, os
print(f"{'run':<34}{'特征':<14}{'生成':>8}{'GT':>8}{'生成/GT':>9}")
for p in sorted(glob.glob("assets/t2_*.json")):
    rs = json.load(open(p, encoding="utf-8"))
    by = {}
    for r in rs:
        gen = "生成图" in r["tag"]
        feat = "style" if "style" in r["tag"] else "dino"
        by.setdefault(feat, {})["gen" if gen else "gt"] = r.get("acc_centroid")
    for feat, d in by.items():
        g, t = d.get("gen"), d.get("gt")
        ratio = (g / t) if (g is not None and t) else None
        f = lambda v: f"{v:>8.4f}" if isinstance(v, float) else f"{'n/a':>8}"
        rr = f"{ratio:>9.2f}" if ratio is not None else f"{'n/a':>9}"
        print(f"{os.path.basename(p)[3:-5]:<34}{feat:<14}{f(g)}{f(t)}{rr}")
print()
print("判据：留字 CV + 最近质心；随机基线 = 1/书家数 ≈ 0.022")
print("      **生成/GT ≥ 0.5** 才算风格真的进了像素；只看生成绝对值会被天花板误导")
PYEOF
