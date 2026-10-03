#!/usr/bin/env bash
# run_d1_visibility.sh — D1：风格条件在 adaLN 里的可见性（真实 ckpt）
#
# 目的：回答"风格向量是不是被 timestep 淹没了"。
#   若 ‖y_emb‖/‖t_emb‖ ≪ 0.3 或 delta_mod(风格轴) < 0.05，
#   说明风格传不进去 —— 那么 S2 的三条新通路也会白搭（都建立在同一个风格向量上）。
#
# 产物：assets/d1_<tag>.json（含逐层逐组的 Δ）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
mkdir -p assets/d1_logs

run_one () {
  local tag="$1" ckpt="$2" cfg="$3" n="${4:-32}"
  if [ ! -f "$ckpt" ]; then echo "[skip] $tag: ckpt 不存在 $ckpt"; return; fi
  if [ ! -f "$cfg" ];  then echo "[skip] $tag: config 不存在 $cfg";  return; fi
  echo "=================================================================="
  echo "[D1] $tag"
  echo "     ckpt=$ckpt"
  echo "=================================================================="
  $PY tools/probe_style_visibility.py \
      --ckpt "$ckpt" --config "$cfg" --n "$n" --device cuda \
      --out "assets/d1_${tag}.json" 2>&1 | tee "assets/d1_logs/${tag}.log"
}

R=assets/results
run_one v13_base_155k   $R/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt        src/train/configs/v13_base_50k.json
run_one v13_wd01_125k   $R/v13_wd01/20260918-210256-v13-base-50k/checkpoints/0125000.pt            src/train/configs/v13_base_50k_wd01.json
run_one v13_12ch_225k   $R/v13_12ch_post/20260918-082201-v13-12ch-post/checkpoints/0022500.pt     src/train/configs/v13_12ch_post.json
run_one v14_s2_160k     $R/v14_style87_s2/20260918-234236-v14-style87-stage2/checkpoints/0160000.pt src/train/configs/v14_style87_stage2.json
run_one v15a_150k       $R/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt src/train/configs/v15a_multistyle_k4_pool.json
run_one v15b_supcon_70k $R/v15b_supcon/20260921-132816-v15b-multistyle-k4-ca/checkpoints/0070000.pt src/train/configs/v15b_supcon.json
run_one v15c_fixed_210k $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt src/train/configs/v15c_fixed.json

echo
echo "==================== 汇总 ===================="
$PY - <<'PYEOF'
import glob, json, os
rows = []
for p in sorted(glob.glob("assets/d1_*.json")):
    d = json.load(open(p, encoding="utf-8"))
    rows.append((os.path.basename(p)[3:-5], d.get("ratio_y_over_t"),
                 d.get("delta_mod_mean"), d.get("delta_mod_layer_imbalance"),
                 d.get("degenerate")))
if not rows:
    print("（没有结果）")
else:
    print(f"{'tag':<22}{'y/t':>9}{'dmod':>10}{'层不平衡':>10}  退化?")
    for t, r, dm, imb, dg in rows:
        def f(v, w):
            return f"{'n/a':>{w}}" if v is None else f"{v:>{w}.4f}"
        print(f"{t:<22}{f(r,9)}{f(dm,10)}{f(imb,10)}  {'是' if dg else '否'}")
    print()
    print("判据: y/t 应在 0.3~3；Δmod(风格轴) 应 > 0.05；层不平衡越接近 1 越好")
PYEOF
