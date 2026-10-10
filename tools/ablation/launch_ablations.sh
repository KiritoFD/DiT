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
#   RESUME=1 ARMS="sym3x_naiveOT noaug_noOT" bash tools/ablation/launch_ablations.sh run
#       ↑ 从各臂**最新 ckpt** 续跑（--resume-full, 恢复 optimizer/scheduler/step）
#
# 日志: $ROOT/logs/ablation/<arm>.log ；结果: assets/results/ablation/<arm>/<ts>/
#
# ★ 2026-10-11 改动（与 4090 同一份代码, 行为靠**配置项**区分）:
#   1) 支持 RESUME=1：按臂找最新 ckpt 传给 --resume-full（避免重启丢进度）
#   2) OMP_NUM_THREADS 默认 8：24 核机器上 preload_workers=16 + workers=8 + torch intra-op
#      会造成运行队列 r≈26 的超额订阅（实测），限制 intra-op 线程减少上下文切换
#   3) 各臂 config 里显式写 "repa_need_image": "never" —— 有 DINO 特征缓存时不载原图
#      （省 15.3 GB RAM + 每步 100 MB H2D；真 miss 会由 REPALoss 显式报错）
set -u

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# ★ 2026-10-11: python 解释器**自动探测** —— 两机路径不同（4090: /opt/conda/envs/cu121,
#   48: /home/ds/miniconda3/envs/pytorch）。原来硬编码 4090 的路径，导致在 48 上
#   **不显式传 PY 就每个臂秒退**（`No such file or directory`）。带 PY= 调用仍可覆盖。
if [ -z "${PY:-}" ]; then
  for _cand in /opt/conda/envs/cu121/bin/python /home/ds/miniconda3/envs/pytorch/bin/python; do
    [ -x "$_cand" ] && { PY="$_cand"; break; }
  done
  PY="${PY:-python3}"
fi
echo "[py] 使用解释器: $PY"
SES="${SES:-ablations}"
LOGDIR="$ROOT/logs/ablation"
ARMS_DEFAULT="noaug_c2ot sym3x_noOT sym3x_naiveOT noaug_noOT thick_c2ot thin_c2ot symwide_c2ot sym4_c2ot"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# 注: 不设 PYTORCH_CUDA_ALLOC_CONF —— train.py 顶部注释说明 expandable_segments 会让
# memory_reserved() 每步波动，v68 的 batch 标定表就是在默认(关闭)下测的，需保持一致。

mkdir -p "$LOGDIR"

resume_args() {
  # 找该臂最新 ckpt -> --resume-full <path>（RESUME=1 时生效）
  # ⚠ 只认真正的 run 目录 `<ts>-abl-<arm>`：结果目录里还有 _active_ckpt_dir.txt 等文件,
  #   用 `ls | tail -1` 会把它当目录 -> 找不到 ckpt（2026-10-11 踩过）。
  [ "${RESUME:-0}" = "1" ] || { echo ""; return; }
  local arm="$1" d ck
  d="$(ls -1d "$ROOT/assets/results/ablation/$arm"/*-abl-*/ 2>/dev/null | sort | tail -1)"
  if [ -z "$d" ]; then
    d="$(ls -1d "$ROOT/assets/results/ablation/$arm"/*/ 2>/dev/null | while read -r x; do
           [ -d "$x/checkpoints" ] && echo "$x"; done | sort | tail -1)"
  fi
  [ -z "$d" ] && { echo ""; return; }
  ck="$(ls -1 "$d"checkpoints/*.pt 2>/dev/null | sort | tail -1)"
  if [ -n "$ck" ]; then echo "--resume-full $ck"; else echo ""; fi
}

status() {
  for arm in ${ARMS:-$ARMS_DEFAULT}; do
    res="$ROOT/assets/results/ablation/$arm"
    last="$(ls -1 "$res" 2>/dev/null | tail -1)"
    ck="none"
    [ -n "$last" ] && ck="$(ls -1 "$res/$last"/checkpoints/*.pt 2>/dev/null | tail -1 | xargs -r basename)"
    printf "%-18s %-34s %-14s %s\n" "$arm" "${last:-none}" "$ck" "$LOGDIR/$arm.log"
  done
}

run() {
  cd "$ROOT" || exit 1
  echo "ROOT=$ROOT PY=$PY OMP_NUM_THREADS=$OMP_NUM_THREADS RESUME=${RESUME:-0}"
  for arm in ${ARMS:-$ARMS_DEFAULT}; do
    cfg="$ROOT/src/train/configs/ablation/$arm.json"
    log="$LOGDIR/$arm.log"
    if [ ! -f "$cfg" ]; then echo "[skip] 配置缺失: $cfg (先跑 make_ablation_configs.py)"; continue; fi
    RES="$(resume_args "$arm")"
    echo "=== [$(date '+%F %T')] RUN $arm  ${RES:+[resume] $RES} ==="
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS="$OMP_NUM_THREADS" \
      "$PY" -u -m src.train.train --config "$cfg" $RES 2>&1 | tee -a "$log"
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
    tmux new-session -d -s "$SES" \
      "ROOT='$ROOT' PY='$PY' ARMS='${ARMS:-}' RESUME='${RESUME:-0}' OMP_NUM_THREADS='$OMP_NUM_THREADS' \
       bash '$ROOT/tools/ablation/launch_ablations.sh' run-inline"
    echo "已启动 tmux 会话 '$SES'。查看: tmux attach -t $SES ；进度: $0 status"
    ;;
  *) echo "用法: $0 [run|run-inline|status]"; exit 2 ;;
esac
