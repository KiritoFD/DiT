#!/usr/bin/env bash
# =============================================================================
# run_skelnet_dit_arms.sh — 骨架生成 DiT 的设计变量串行消融 (~12h)
#
# 任务: (标准骨架 g_std, 风格 e) -> 该书家真迹的骨架 latent (7px 载体)。
#       与 warp 头的区别: 这里不重采样任何东西, 从噪声直接生成目标。
#
# 判据: 容差版 clDice(tol=4px2) 在**按字符留出**的验证集上。
#       ⚠ 必须带容差 —— 精确像素重合对 1px 中心线不可用:
#         把输入标准骨架直接当预测, 精确 clDice 只有 0.045, 容差 4px 才是 0.325。
#       所以门槛 = clDice > 0.325 (copy baseline)。
#
# 六个单变量臂:
#   A_base   基线            7px / d6 h256 / inject2 / gs0.6 / wd0.03 / flow-v
#   B_w3     目标载体 3px    (3px 在 32x32 上只有 0.375 格 = 亚像素)
#   C_inj4   骨架注入 2->4   (std 是未见字唯一的字身份来源)
#   D_x0     参数化 v->x0    (配对近似确定性, 可能收敛更快)
#   E_small  容量 d4/h192    (约 4M, 验证"参数更小够不够")
#   F_wd     正则 wd 0.03->0.1
#
# 特性: 串行; 每臂独立 log / ckpt; 失败记录 return code 但不中断队列;
#       已有 .done 标记的臂自动跳过 (可断点续跑)。
#
# 用法: bash _sync_work/run_skelnet_dit_arms.sh
#        STEPS=2000 bash _sync_work/run_skelnet_dit_arms.sh   # 探针
# =============================================================================
set -u
cd /root/Workspace/xy/DiT || exit 1
mkdir -p logs/skelnet_dit_arms

STEPS=${STEPS:-40000}
BATCH=${BATCH:-128}
LR=${LR:-2e-4}
EVAL=${EVAL:-1000}
SAVE=${SAVE:-500}
VALN=${VALN:-128}
SAMP=${SAMP:-20}
ES=${ES:-10}
RUNS_DIR=_sync_work/skelnet_dit_arms
mkdir -p "$RUNS_DIR"
SUMMARY="$RUNS_DIR/summary.csv"
[ -f "$SUMMARY" ] || echo "arm,steps,best_cldice,best_step,rc,elapsed_min" > "$SUMMARY"

# 清掉残留训练, 避免抢卡
for p in $(ps -eo pid,cmd | awk '/train_skelnet_dit/ && !/awk/ {print $1}'); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 5

echo "=============================================================="
echo " SkelNet-DiT 设计变量串行消融  $(date '+%F %T')"
echo " STEPS=$STEPS BATCH=$BATCH LR=$LR EVAL=$EVAL VALN=$VALN SAMP=$SAMP ES=$ES"
echo " 预计: 6 臂 x (40k/8 步每秒 + 评测) ~= 11-12 h"
echo " $(nvidia-smi -L | head -1)"
echo "=============================================================="

run_arm () {
  local name="$1"; shift
  local extra=("$@")
  local log="logs/skelnet_dit_arms/${name}.log"
  local out="assets/skelnet_dit_${name}.pt"
  local done_marker="$RUNS_DIR/${name}.done"
  if [ -f "$done_marker" ]; then
    echo ">>> [$(date '+%T')] $name 已完成 (有 .done), 跳过"
    return 0
  fi
  echo ""
  echo "--------------------------------------------------------------"
  echo ">>> [$(date '+%F %T')] $name  额外参数: ${extra[*]:-（无）}"
  echo "--------------------------------------------------------------"
  local t0=$(date +%s)
  /opt/conda/envs/cu121/bin/python tools/train_skelnet_dit.py \
      --steps "$STEPS" --batch "$BATCH" --lr "$LR" \
      --eval-every "$EVAL" --save-every "$SAVE" \
      --val-n "$VALN" --sample-steps "$SAMP" --es-patience "$ES" \
      --log "$log" --out "$out" "${extra[@]}" \
      > "${log}.stdout" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local mins=$(( (t1 - t0) / 60 ))
  # 从 log 里取最佳 clDice 与其步数
  local best beststep
  best=$(grep -a '新最佳 clDice' "$log" 2>/dev/null | tail -1 | sed 's/.*clDice //')
  beststep=$(grep -a -B1 '新最佳 clDice' "$log" 2>/dev/null | grep -a '\[eval\]' | tail -1 | sed 's/.*step \([0-9]*\).*/\1/')
  echo "$name,$STEPS,${best:-NA},${beststep:-NA},$rc,$mins" >> "$SUMMARY"
  echo ">>> [$(date '+%T')] $name 结束 rc=$rc 用时 ${mins}min  best_clDice=${best:-NA}"
  if [ "$rc" -eq 0 ]; then
    touch "$done_marker"
  else
    echo "    [!] rc=$rc 非 0 —— 看 ${log}.stdout 尾部; 队列继续"
    tail -5 "${log}.stdout" | sed 's/^/    /'
  fi
}

# ── 六个臂 ────────────────────────────────────────────────────────────────
run_arm A_base
run_arm B_w3     --tgt-shards data/top10_style23/shards_gtskel_w3
run_arm C_inj4   --inject-layers 4
run_arm D_x0     --pred x0
run_arm E_small  --depth 4 --hidden 192 --heads 4
run_arm F_wd     --wd 0.1

echo ""
echo "=============================================================="
echo " 全部结束  $(date '+%F %T')"
echo "=============================================================="
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
