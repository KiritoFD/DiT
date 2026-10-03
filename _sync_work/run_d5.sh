#!/usr/bin/env bash
# D5 - 条件嵌入本身的书家可分性（只读权重，CPU，零 GPU 占用）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
mkdir -p assets/d5_logs

run_one () {
  local tag="$1" ckpt="$2"
  [ -f "$ckpt" ] || { echo "[skip] $tag: $ckpt"; return; }
  local cfg="$(dirname "$(dirname "$ckpt")")/resolved_config.json"
  echo "=================================================================="
  echo "[D5] $tag"
  echo "=================================================================="
  $PY tools/probe_embed_separability.py --ckpt "$ckpt"       --config "$cfg" --out "assets/d5_${tag}.json" 2>&1       | tee "assets/d5_logs/${tag}.log" | tail -45
}

R=assets/results
run_one v13_base_50k       $R/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt
run_one v13_12ch_post      $R/v13_12ch_post/*/checkpoints/0022500.pt
run_one v13_wd01           $R/v13_wd01/*/checkpoints/0125000.pt
run_one v15a_multistyle_k4 $R/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt
run_one v15b_supcon        $R/v15b_supcon/*/checkpoints/0070000.pt
run_one v15c_fixed         $R/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt
echo 'D5 ALL DONE'
