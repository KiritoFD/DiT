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
# ★ 铁律: 不设 PYTORCH_CUDA_ALLOC_CONF (不允许 expandable_segments)。这里主动清掉,
#   防止 shell/父进程带进来 (train.py 顶部还有一道代码闸门, 双保险)。
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
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
  # 可选: A 路复用上一轮已有 ckpt 续跑 (省时间; B 路必须从零, 因为注入方式不同)
  if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
    local _st
    _st=$(basename "$RESUME_CKPT" .pt | sed 's/^0*//')
    ARGS+=(--resume-full "$RESUME_CKPT" --resume-lr "${RESUME_LR:-5e-5}")
    echo "[resume] $RESUME_CKPT (step=$_st) -> 总 100000, 起始 lr=${RESUME_LR:-5e-5}"
  fi
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
  # ⚠ train.py **OOM 也退出码 0** (实测: 报 OutOfMemoryError 后仍打印 Done! 退出 0) ->
  #   不能只看 rc, 必须查日志里有没有 OOM/Traceback, 否则会静默丢失一整路。
  if grep -aq 'OutOfMemoryError\|CUDA out of memory\|Traceback (most recent call last)' "$log"; then
    echo "[FATAL] $name 日志里出现 OOM/Traceback (rc=$rc 不可信)"
    rc=9
  fi
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
RESUME_CKPT=""            # ★ B 路必须从零训练 (注入方式变了, resume 不公平)

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
