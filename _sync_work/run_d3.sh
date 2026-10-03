#!/usr/bin/env bash
# D3 - 风格差异的空间分布（固定字+固定噪声，只换书家）
# ★ 2026-09-22 修正：改用 strict 集（skel 覆盖率 100%）+ 逐级过滤
#   之前用训练 csv 会撞 make_eval_cache 的 98% skel 闸（训练 id != eval shard id）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python
mkdir -p assets/d3_logs

run_one () {
  local tag="$1" ckpt="$2" mode="${3:-any}"
  [ -f "$ckpt" ] || { echo "[skip] $tag: $ckpt"; return; }
  echo "=================================================================="
  echo "[D3] $tag  script-mode=$mode"
  echo "=================================================================="
  $PY tools/probe_style_spatial.py --ckpt "$ckpt"       --eval-csv assets/eval_v13_strict.csv --cfg 1.0       --n-chars 7 --n-calligs 2 --steps 50 --script-mode "$mode"       --out "assets/d3_${tag}_${mode}.json" 2>&1 |       tee "assets/d3_logs/${tag}_${mode}.log" | grep -vE 'Warning|warn|pkg_resources|_pytree' | tail -28
}

R=assets/results
run_one v13_base_50k       $R/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt any
run_one v13_12ch_post      $R/v13_12ch_post/*/checkpoints/0022500.pt any
run_one v13_wd01           $R/v13_wd01/*/checkpoints/0125000.pt any
run_one v15a_multistyle_k4 $R/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt any
run_one v15c_fixed         $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt any
# 纯书家对照（同书体，排除书体混杂）
run_one v13_base_50k_same       $R/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt same
run_one v15c_fixed_same         $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt same
echo 'D3 ALL DONE'
