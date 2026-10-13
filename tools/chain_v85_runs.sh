#!/bin/bash
# v85 收尾自动化：① 等 v85A 跑完 ② 起 Control ③ 编码器过门槛后起 v85B
# 说明: GPU 只有 24G, v85A 占 22.47G ⇒ 必须**顺序**执行（并行会 OOM ✗ 且撞 DDP 端口 ✓）
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.

wait_v85a() {
  local L
  L=$(ls -t logs/v85A_run_*.log | head -1)
  echo "[wait] v85A 日志 $L"
  while :; do
    s=$(grep -oE '\(step=0*[0-9]+\)' "$L" | tail -1 | tr -dc '0-9')
    [ -z "$s" ] && s=0
    if [ "$s" -ge 30000 ] || ! pgrep -f 'v85A_style_ada.json' >/dev/null; then
      echo "[wait] v85A 结束（最后 step=$s ✓）"
      break
    fi
    sleep 120
  done
}

launch() {  # $1=cfg $2=port
  local cfg=$1 port=$2 L
  L="logs/v85_${cfg}_$(date +%Y%m%d-%H%M%S).log"
  echo "[launch] $cfg (MASTER_PORT=$port) -> $L"
  MASTER_PORT=$port nohup $PY -m src.train.train --config "src/train/configs/$cfg.json" \
      --num-workers 4 > "$L" 2>&1 &
  echo "$L" > "/tmp/v85_${cfg}.logpath"
  sleep 90
  grep -E 'warm-start|style-ada|style-rank|\[optim\]|epoch\]' "$L" | tail -6 | sed 's/^/    /'
}

wait_v85a
echo "=== 起 Control（同 LR 1e-5 续训 30k, 无 adapter ✓ 方案要求的对照 ✓）==="
# ⚠ 坑: v85A 用的是**默认端口 12355** ✗ ⇒ 接力若也用 12355 会 EADDRINUSE 直接死掉 ✓
#   ⇒ 每次启动都必须换端口 ✓（实测: Control 曾在 06:09 因此启动即失败 ✗）
launch v85_control 12357
