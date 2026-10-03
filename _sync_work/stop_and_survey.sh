#!/bin/bash
# 停训练 + 盘点: 日志/seen集/全部ckpt家底 (为"重eval + seen集 + ckpt轨迹"做准备)
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 训练进程 ##########"
ps -eo pid,etime,cmd | grep -E 'train\.py|run_stdmix|purestd' | grep -v grep
echo "--- kill ---"
pkill -f 'src/train/train.py' && echo "killed train.py" || echo "no train.py running"
sleep 8
echo "--- after ---"
n=$(ps -eo cmd | grep -E 'train\.py' | grep -v grep | wc -l)
echo "剩余 train.py 进程 = $n"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "########## 2. 最近的训练日志 ##########"
ls -lat _sync_work/*.log logs/*.log 2>/dev/null | head -6
for f in $(ls -t _sync_work/*.log 2>/dev/null | head -2); do
  echo "--- $f ---"; tail -5 "$f"
done

echo
echo "########## 3. 历史 seen 集 / eval 集 ##########"
ls -la assets/eval_*.csv 2>/dev/null
for f in assets/eval_seen_v10.csv assets/eval_v13_seen.csv assets/eval_seen20.csv \
         assets/eval_top10_real_200.csv exp-std/csv/eval200.csv; do
  if [ -f "$f" ]; then
    echo "--- $f ($(( $(wc -l < "$f") - 1 )) 数据行) ---"
    head -2 "$f"
  fi
done

echo
echo "########## 4. ckpt 家底(按目录计数) ##########"
for d in assets/results _archive/20261003_twostage exp exp-std/ckpt; do
  if [ -d "$d" ]; then
    c=$(find "$d" -name '*.pt' 2>/dev/null | wc -l)
    s=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "$d : $c 个 .pt, $s"
  fi
done

echo
echo "########## 5. 顶层 run 目录(含 step 跨度) ##########"
find assets/results exp _archive/20261003_twostage -maxdepth 2 -name 'checkpoints' -type d 2>/dev/null \
  | while read -r ck; do
      n=$(ls "$ck"/*.pt 2>/dev/null | wc -l)
      first=$(ls "$ck"/*.pt 2>/dev/null | head -1 | xargs -r basename)
      last=$(ls "$ck"/*.pt 2>/dev/null | tail -1 | xargs -r basename)
      printf '%-88s n=%-4s %s .. %s\n' "$ck" "$n" "$first" "$last"
    done | head -60
