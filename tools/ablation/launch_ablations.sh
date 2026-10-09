#!/usr/bin/env bash
# launch_ablations.sh — 在单卡机器上顺序跑「OT × 数据增强」消融臂。
#
# 用 v68 的配方做 base（Sp/2, lr 5e-5, cosine, w_repa 0.03, eval200fix），
# 每个臂只改「数据增强 / OT」，并把 batch*steps 对齐 v68 的 57.6M 总样本预算。
#
# 用法（在 48 上，仓库根目录）:
#   bash tools/ablation/launch_ablations.sh status        # 看每臂进度
#   bash tools/ablation/launch_ablations.sh run           # tmux 后台顺序跑
#   bash tools/ablation/launch_ablations.sh run-inline    # 前台跑（调试）
#   ARMS="noaug_c2ot sym3x_noOT" bash tools/ablation/launch_ablations.sh run   # 只跑子集
#
# 日志: $ROOT/logs/ablation/<arm>.log ；结果: assets/results/ablation/<arm>/<ts>/
set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/opt/conda/envs/cu121/bin/python}"
SES="${SES:-ablations}"
LOGDIR="$ROOT/logs/ablation"
ARMS_DEFAULT="noaug_c2ot sym3x_noOT sym3x_naiveOT noaug_noOT thick_c2ot thin_c2ot symwide_c2ot sym4_c2ot"
# 注: 不设 PYTORCH_CUDA_ALLOC_CONF —— train.py 顶部注释说明 expandable_segments 会让
# memory_reserved() 每步波动，v68 的 batch 标定表就是在默认(关闭)下测的，需保持一致。

mkdir -p "$LOGDIR"

status() {
  for arm in ${ARMS:-$ARMS_DEFAULT}; do
    res="$ROOT/assets/results/ablation/$arm"
    last="$(ls -1 "$res" 2>/dev/null | tail -1)"
    ck="none"
    [ -n "$last" ] && ck="$(ls -1 "$res/$last"/*.pt 2>/dev/null | tail -1 | xargs -r basename)"
    printf "%-18s %-14s %-22s %s\n" "$arm" "${last:-none}" "$ck" "$LOGDIR/$arm.log"
  done
}

run() {
  cd "$ROOT" || exit 1
  echo "ROOT=$ROOT PY=$PY"
  for arm in ${ARMS:-$ARMS_DEFAULT}; do
    cfg="$ROOT/src/train/configs/ablation/$arm.json"
    log="$LOGDIR/$arm.log"
    if [ ! -f "$cfg" ]; then echo "[skip] 配置缺失: $cfg (先跑 make_ablation_configs.py)"; continue; fi
    echo "=== [$(date '+%F %T')] RUN $arm ($cfg) ==="
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      "$PY" -u -m src.train.train --config "$cfg" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "=== [$(date '+%F %T')] DONE $arm rc=$rc ==="
  done
  echo "=== [$(date '+%F %T')] ALL ARMS FINISHED ==="
}

case "${1:-run}" in
  status) status ;;
  run-inline) run ;;
  run)
    if tmux has-session -t "$SES" 2>/dev/null; then
      echo "tmux 会话 '$SES' 已存在；先 tmux attach -t $SES 或 tmux kill-session -t $SES"
      exit 1
    fi
    # 训练须在默认 allocator 配置下跑 (v68 的 batch 标定即默认, 不设该变量),
    # 故这里不再转发 PYTORCH_CUDA_ALLOC_CONF。
    tmux new-session -d -s "$SES" \
      "ROOT='$ROOT' PY='$PY' ARMS='${ARMS:-}' bash '$ROOT/tools/ablation/launch_ablations.sh' run-inline"
    echo "已启动 tmux 会话 '$SES'。查看: tmux attach -t $SES ；进度: $0 status"
    ;;
  *) echo "用法: $0 [run|run-inline|status]"; exit 2 ;;
esac
