#!/bin/bash
# 睡到 ETA 附近再看编码进度与速度
cd /root/Workspace/xy/DiT || exit 1
L=exp-std/logs_purestd/flux_pipeline_latest.log
SLEEP="${1:-420}"
echo "[wait] sleep ${SLEEP}s  现在 $(date '+%F %T')"
sleep "$SLEEP"

echo
echo "########## $(date '+%F %T') 日志尾 ##########"
tail -20 "$L" 2>/dev/null

echo
echo "########## 编码阶段汇总 ##########"
grep -nE '\[in\]|\[shard\]|\[done\]|img/s|########|全量编码|竞技场|判据|guard|lat\]|auroc|inv\]' "$L" 2>/dev/null | tail -40

echo
echo "########## 速度(last 20 个采样点) ##########"
grep -oE '[0-9]+ img/s' "$L" 2>/dev/null | tail -20 | tr '\n' ' '
echo

echo
echo "########## GPU ##########"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader

echo
echo "########## 产物 ##########"
for d in exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16 exp-std/signal_arena_full; do
  if [ -d "$d" ]; then
    echo "--- $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
    ls "$d" 2>/dev/null | tail -4
  fi
done

echo
echo "########## 进程 ##########"
ps -eo pid,etime,pcpu,cmd | grep -E 'encode_flux|latent_signal' | grep -v grep | head -4
