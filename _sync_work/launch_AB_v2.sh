#!/usr/bin/env bash
# A/B 启动器 v2 —— 补 P1-3: 每路启动前 kill 残留进程 + 校验 GPU 空闲 + 环境变量铁律。
#
# ⚠ 为什么不直接改 launch_AB_100k.sh: 它**正在被 bash 执行** (A 路在跑),
#   就地编辑运行中的 bash 脚本可能让解释器从错误的字节偏移继续读 -> 行为未定义。
#   所以新开 v2, 下一次启动用 v2。
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

# ─────────────── P1-3 启动护栏 ───────────────
guard() {
  local tag="$1"
  echo "[guard:$tag] ── 启动前检查 ──"
  # 1) 铁律: 不许 expandable_segments (train.py 内还有一道代码闸门, 双保险)
  unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
  if env | grep -q 'PYTORCH_CUDA_ALLOC_CONF'; then
    echo "[FATAL] 环境里仍有 PYTORCH_CUDA_ALLOC_CONF, 拒绝启动"
    return 6
  fi
  echo "[guard:$tag] ✓ 无 PYTORCH_CUDA_ALLOC_CONF"
  # 2) kill 残留 train.py (0 个也允许)
  if pgrep -f 'src/train/train.py' >/dev/null 2>&1; then
    echo "[guard:$tag] ⚠ 发现残留 train.py, 先 kill:"
    pgrep -af 'src/train/train.py' | head -3 | sed 's/^/        /'
    pkill -f 'src/train/train.py'
    sleep 10
    if pgrep -f 'src/train/train.py' >/dev/null 2>&1; then
      echo "[FATAL] kill 之后仍有残留, 拒绝启动 (避免两进程抢卡)"
      return 7
    fi
  fi
  echo "[guard:$tag] ✓ 无残留 train.py"
  # 3) 校验 GPU 空闲 (阈值 2000 MiB: 只容忍 CUDA context 级别残留)
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  used=${used:-999999}
  if [ "$used" -gt 2000 ] 2>/dev/null; then
    echo "[FATAL] GPU 仍占用 ${used} MiB (>2000), 拒绝启动"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/        /'
    return 8
  fi
  echo "[guard:$tag] ✓ GPU 空闲 (${used} MiB)"
  return 0
}

run_one() {
  local cfg="$1" name="$2" log="$3"
  local ARGS=(-u src/train/train.py --config "$cfg"
              --experiment-name "$name"
              --results-dir "$RUNS"
              --global-batch-size 128
              --skel-latent-shards-weights 1.0,0.0
              --max-steps 100000
              --lr 5e-5)
  if [ -n "${RESUME_CKPT:-}" ] && [ -f "${RESUME_CKPT}" ]; then
    local _st
    _st=$(basename "$RESUME_CKPT" .pt | sed 's/^0*//')
    ARGS+=(--resume-full "$RESUME_CKPT" --resume-lr "${RESUME_LR:-5e-5}")
    echo "[resume] $RESUME_CKPT (step=$_st) -> 总 100000, 起始 lr=${RESUME_LR:-5e-5}"
  fi
  guard "$name" || return $?
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
  # ⚠ train.py **OOM 也退出码 0** (实测: 报 OutOfMemoryError 后仍打印 Done!)
  if grep -aq 'OutOfMemoryError\|CUDA out of memory\|Traceback (most recent call last)' "$log"; then
    echo "[FATAL] $name 日志出现 OOM/Traceback (rc=$rc 不可信)"
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
echo "A/B 两路 100k (v2: 含启动护栏)  产物=$RUNS  日志=$LOGD"
echo "================================================================"
TS=$(date +%Y%m%d-%H%M%S)
echo ""; echo "############ [A] 分工注入 (骨架 xattn + 风格 adaLN) ############"
run_one src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
        "v50-A-space-xattn-style-adaln-top10" "$LOGD/A_${TS}.log" || exit $?
echo "############ [A] 完成 -> [B] ############"
RESUME_CKPT=""
TS=$(date +%Y%m%d-%H%M%S)
echo ""; echo "############ [B] 联合 KV (256骨架token ⊕ K=4风格token, PCA+tau0.1+ortho) ############"
run_one src/train/configs/v51_B_joint_kv_k4_top10.json \
        "v51-Bp-joint-kv-k4-pca-tau010-ortho" "$LOGD/B_${TS}.log" || exit $?
echo ""; echo "===== A/B 两路完成 ====="
