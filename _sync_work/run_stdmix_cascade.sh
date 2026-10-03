#!/usr/bin/env bash
# run_stdmix_cascade.sh — 条件采样课程级联 (top10 / 23 槽位 / adaln×4 / 单阶段 std 条件模型)
#
# 机制 (仓库原有, 未改模型代码): 数据集按 skel_latent_shards_weights **每样本随机二选一**
#   —— std 标准字骨架 (部署条件) 或 GT 真迹骨架 (等于给答案)。课程: 从大部分 GT 起步
#   (先学"出墨"), 最后完全 std (= 推理时的真实条件)。
#
# 形态: 级联 —— 每段跑到收敛/步数上限 -> 存 ckpt -> 下一段接着跑。理由: 每段一个可判分点
#   (同一把尺子: 全都在部署条件 std 下评 eval200 held-out), 条件分布不持续移动,
#   每段可回滚; 成本≈连续。
#
# ════════════ 踩过的坑, 全部已焊闸门 (2026-10-03) ════════════
#  [A] `--fresh-scheduler` 是 type=_str_to_bool -> 裸写会 argparse 报错秒退。
#      现在: 每段启动前 PREFLIGHT 用**真实 CLI 解析器**干跑 argv, 报错就不启动。
#  [B] ckpt 名零填充 (0017500.pt): bash 算术把前导 0 当**八进制** -> 步数算错。
#      现在: dec_step() 先剥前导 0, 且断言 STEP>0 (解析失败立即退出)。
#  [C] `--fresh-scheduler` 的真实语义是**步数计数器归零**(实测日志 "步数计数器归零
#      (原 0 -> 0)"), 会让 max-steps 的绝对语义与 early-stop-from-step 的绝对过滤失效,
#      并与共享 summary.csv 的 step 冲突。现在: 明令禁用 + 自检拦截 + preflight 兜底。
#  [D] 上一版静默死 7 小时没人发现 -> 现在失败会写 exp-std/logs/CASCADE_FAILED.txt。
# ═══════════════════════════════════════════════════════════
set -u
ROOT=/root/Workspace/xy/DiT
cd "$ROOT" || exit 1
PY=${PY:-/opt/conda/envs/cu121/bin/python}
CFG=${CFG:-src/train/configs/v46_std_adaln4_top10.json}
RUNS=${RUNS:-exp-std/runs}
LOGD=${LOGD:-exp-std/logs}
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/torch/inductor}
# 铁律: 不设 PYTORCH_CUDA_ALLOC_CONF (不允许 expandable_segments)
mkdir -p "$LOGD" "$RUNS"

# ---- 可调参数 ----------------------------------------------------------
STAGES=${STAGES:-"0.2 0.4 0.6 0.8 1.0"}   # p_std 阶梯, 最后必须是 1.0 (部署点)
STAGE_STEPS=${STAGE_STEPS:-30000}          # 每段最大步数 (段内早停可能提前收工)
FIRST_LR=${FIRST_LR:-5e-5}                 # 第一段 lr (从零)
STAGE_LR=${STAGE_LR:-2e-5}                 # 后续段 lr (接续微调)
BATCH=${BATCH:-384}                        # 单阶段很轻: 256 只吃 14G, 384 ~20G
INIT=${INIT:-}                             # 起点 ckpt; 空 = 从零
DRY=${DRY:-0}          # 1 = 只打印每段 argv(ckpt 链按算术模拟), 不启训练
PREFLIGHT=${PREFLIGHT:-1}   # 1 = 每段启动前用真实 CLI 解析器干跑校验 argv (强烈建议保持 1)
# ------------------------------------------------------------------------

# 零填充步数 -> 十进制 (bash 算术会把 0017500 当八进制! 见坑 B)
dec_step() { echo "$1" | sed 's/[^0-9]//g; s/^0*//'; }

echo "================================================================"
echo "条件采样课程级联  STAGES=[$STAGES]  每段<=$STAGE_STEPS 步  batch=$BATCH"
echo "首段 lr=$FIRST_LR  后续段 lr=$STAGE_LR  起点=${INIT:-<从零>}"
echo "配置 $CFG   DRY=$DRY  PREFLIGHT=$PREFLIGHT"
echo "数据 exp-std/{data,csv}  产物 $RUNS"
echo "================================================================"

# shellcheck disable=SC2086
set -- $STAGES
TOTAL=$#
PREV="$INIT"
i=0
for P in $STAGES; do
  i=$((i + 1))
  GT=$(awk -v p="$P" 'BEGIN{printf "%.4f", 1 - p}')
  if [ "$i" -eq 1 ]; then LR="$FIRST_LR"; else LR="$STAGE_LR"; fi
  TAG="p$P"
  EXP="v46-std-adaln4-top10-$TAG"
  TS=$(date +%Y%m%d-%H%M%S)
  LOG="$LOGD/stage${i}_${TAG}_${TS}.log"

  echo ""
  echo "===== [stage $i/$TOTAL] p_std=$P  p_gt=$GT  lr=$LR  <= $STAGE_STEPS 步 ====="

  ARGS=(-u src/train/train.py --config "$CFG"
        --experiment-name "$EXP"
        --results-dir "$RUNS"
        --global-batch-size "$BATCH"
        --skel-latent-shards-weights "$P,$GT"
        --lr "$LR")

  if [ -n "$PREV" ]; then
    STEP=$(dec_step "$(basename "$PREV" .pt)")
    [ -z "$STEP" ] && STEP=0
    if [ "$STEP" -le 0 ]; then
      echo "[FATAL] 无法从 '$PREV' 解析步数 (得到 '$STEP') —— 拒绝启动, 免得步数算错"
      exit 4
    fi
    MAX=$((STEP + STAGE_STEPS))
    ARGS+=(--resume-full "$PREV" --resume-lr "$LR" --max-steps "$MAX"
           --early-stop-from-step "$((STEP + 1))")
    echo "     接续 $PREV (step=$STEP) -> 绝对 step $MAX | 早停只看 step >= $((STEP + 1))"
    NEXT_CKPT="$RUNS/${TS}-${EXP}/checkpoints/$(printf '%07d' "$MAX").pt"
  else
    ARGS+=(--max-steps "$STAGE_STEPS" --early-stop-from-step 1)
    echo "     从零训练, 上限 $STAGE_STEPS 步"
    NEXT_CKPT="$RUNS/${TS}-${EXP}/checkpoints/$(printf '%07d' "$STAGE_STEPS").pt"
  fi

  # ---- 自检: 禁用 --fresh-scheduler (坑 C) ----
  case " ${ARGS[*]} " in
    *"--fresh-scheduler"*)
      echo "[FATAL] argv 里出现 --fresh-scheduler: 它的语义是步数计数器归零, 会破坏"
      echo "        max-steps 绝对语义 / early-stop-from-step 绝对过滤。已拒绝启动。"
      exit 5
      ;;
  esac

  # ---- 干跑模式 ----
  if [ "$DRY" = "1" ]; then
    printf '     argv: %q ' "$PY"
    printf '%q ' "${ARGS[@]}"
    echo ""
    PREV="$NEXT_CKPT"
    continue
  fi

  # ---- 启动前预检: 真实 CLI 解析器干跑 (坑 A/C) ----
  if [ "$PREFLIGHT" = "1" ]; then
    if ! $PY tools/preflight_stage_args.py "${ARGS[@]}"; then
      echo "[FATAL] 阶段 $i argv 预检未通过 -> 不启动 (避免重复 stage 2 那种秒退)"
      { echo "preflight_failed stage=$i p_std=$P time=$(date '+%F %T')"
        printf '%q ' "$PY"; printf '%q ' "${ARGS[@]}"; echo; } \
        > "$LOGD/CASCADE_FAILED.txt"
      exit 3
    fi
  fi

  $PY "${ARGS[@]}" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "----- [stage $i] 退出码 $RC, 日志 $LOG -----"
  if [ "$RC" -ne 0 ]; then
    echo "[FATAL] 阶段 $i 失败 (rc=$RC), 级联中止。修好后重跑本段即可 (起点仍是上一段的 ckpt)。"
    { echo "stage=$i p_std=$P rc=$RC time=$(date '+%F %T')"
      echo "log=$LOG"
      tail -5 "$LOG"; } > "$LOGD/CASCADE_FAILED.txt"
    exit "$RC"
  fi

  PREV=$(ls -t "$RUNS"/*"$TAG"/checkpoints/*.pt 2>/dev/null | head -1)
  if [ -z "$PREV" ]; then
    echo "[FATAL] 阶段 $i 没产出 ckpt (检查 $LOG), 级联中止"
    echo "no_ckpt stage=$i p_std=$P log=$LOG" > "$LOGD/CASCADE_FAILED.txt"
    exit 1
  fi
  echo "[stage $i] 下一段起点 = $PREV"
  echo "[stage $i] 本段判分: $(grep -E 'eval200.*ssim=|strict_pred.*ssim=' "$LOG" | tail -1)"
done

echo ""
echo "=========== 级联完成 ==========="
echo "各段 ckpt: $RUNS/"
echo "判分曲线: 各段日志的 [in-mem-eval] 行 / $RUNS/eval_stdskel_summary.csv"
