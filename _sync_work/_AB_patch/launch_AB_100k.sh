#!/usr/bin/env bash
# 顺序跑 A/B 两路 100k step (单卡 24G, 训练 ~5.7 steps/s @ batch128 -> 每路约 5 小时)
#   A = 分工注入: C_space 走 ZeroCrossAttention(xattn×12), C_style 走 adaLN(单向量 128d)
#   B = 联合 KV : C_space 256 token(+pos) ⊕ C_style K=4 token(+role) 拼成每层 xattn 的 K/V
# 两路唯二共同设定: 纯 std 条件(skel weights 1.0,0.0) / batch 128 / lr 5e-5 cosine / 100k / 同数据同评测集
set -u
ROOT=/root/Workspace/xy/DiT
cd "$ROOT" || exit 1
PY=${PY:-/opt/conda/envs/cu121/bin/python}
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torch/inductor}
RUNS=exp-std/runs_AB
LOGD=exp-std/logs_AB
mkdir -p "$RUNS" "$LOGD"

run_one() {
  local cfg="$1" name="$2" log="$3"
  local ARGS=(-u src/train/train.py --config "$cfg"
              --experiment-name "$name"
              --results-dir "$RUNS"
              --global-batch-size 128
              --skel-latent-shards-weights 1.0,0.0
              --max-steps 100000
              --lr 5e-5)
  echo "[preflight] $name"
  if ! $PY tools/preflight_stage_args.py "${ARGS[@]}"; then
    echo "[FATAL] preflight 未通过: $cfg"
    { echo "preflight_failed $name $(date '+%F %T')"; printf '%q ' "${ARGS[@]}"; echo; } \
      > "$LOGD/AB_FAILED.txt"
    return 3
  fi
  echo "[run] $PY ${ARGS[*]}"
  $PY "${ARGS[@]}" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  echo "----- [$name] 退出码 $rc  日志 $log -----"
  if [ "$rc" -ne 0 ]; then
    { echo "rc=$rc name=$name time=$(date '+%F %T')"; echo "log=$log"; tail -8 "$log"; } \
      > "$LOGD/AB_FAILED.txt"
    return "$rc"
  fi
  return 0
}

echo "================================================================"
echo "A/B 两路 100k  产物=$RUNS  日志=$LOGD"
echo "  A: v50-A-space-xattn-style-adaln-top10"
echo "  B: v51-Bp-joint-kv-k4-pca-tau010-ortho  (PCA正交 + tau=0.1 + 正交正则)"
echo "================================================================"

TS=$(date +%Y%m%d-%H%M%S)
echo ""
echo "############ [A] 分工注入 (骨架 xattn + 风格 adaLN) ############"
run_one src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
        "v50-A-space-xattn-style-adaln-top10" "$LOGD/A_${TS}.log" || exit $?
echo "############ [A] 完成 -> 开始 [B] ############"

TS=$(date +%Y%m%d-%H%M%S)
echo ""
echo "############ [B] 联合 KV (256 骨架token ⊕ K=4 风格token) ############"
run_one src/train/configs/v51_B_joint_kv_k4_top10.json \
        "v51-Bp-joint-kv-k4-pca-tau010-ortho" "$LOGD/B_${TS}.log" || exit $?

echo ""
echo "================================================================"
echo "A/B 两路 100k 全部完成。判分: $RUNS/eval_stdskel_summary.csv"
echo "  对照组: '什么都不做'(std 直出)  ssim 0.5990 / lpips 0.3607 / ink_iou 0.1687 / frag 1.02"
echo "================================================================"
