#!/bin/bash
# D6 条件因果性 —— 正式批量（改进版：16 字 + 中位数判据 + null 复现对照）
#
# 目的：区分「通路失效(a)」vs「表达幅度太小(b)」
#
# 判据（★ 用中位数，均值会被 D_null=0 的极值污染）：
#   ratio_null  = D(换书家)/D(换null)   >>1.5 = 真在区分书家
#                                         ~1  = 只知道"有/无"，具体书家无信息
#   ratio_noise = D(换书家)/D(换噪声)   >1.5 = 书家效应可从随机性中辨识
#   D_null_rep  = 两次相同 null 条件的差异  应 ≈0（判断 null 是否确定性）
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/eval_v13_strict.csv
LOGD=_sync_work/d6_logs
mkdir -p $LOGD

run_one () {
  local TAG=$1 CKPT=$2 NCH=${3:-16} STEPS=${4:-50} CFG=${5:-1.0}
  if [ -f "assets/d6_${TAG}.json" ]; then
    echo "[D6] SKIP $TAG (已存在)"; return
  fi
  echo "[D6] ===== $TAG  n_chars=$NCH steps=$STEPS cfg=$CFG ====="
  $PY tools/probe_cond_causality.py \
      --ckpt "$CKPT" --eval-csv $CSV \
      --cfg $CFG --n-chars $NCH --steps $STEPS \
      --out "assets/d6_${TAG}.json" \
      > "$LOGD/d6_${TAG}.log" 2>&1
  local rc=$?
  echo "[D6] $TAG exit=$rc"
  grep -vE 'Warning|warn|_pytree|pkg_res' "$LOGD/d6_${TAG}.log" \
    | grep -E '均值|中位|复现|✓|✗|◐' | tail -10
}

B=assets/results
# 主对照：v13_base（历史最好 T2）；cfg=1.0 与 0.7 各跑一次（CFG 通路抵消检验）
run_one v13_base_50k        $B/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt 16 50 1.0
run_one v13_base_50k_cfg07  $B/v13_base_50k/20260917-211905-v13-base-50k/checkpoints/0155000.pt 16 50 0.7
# 两个 S2 前置对照：v15a（可训表）与 v15c（表最可分但 dmod 最低）
run_one v15a_multistyle     $B/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt 16 50 1.0
run_one v15c_fixed          $B/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt 16 50 1.0

echo "[D6] ALL DONE"
ls -la assets/d6_*.json
