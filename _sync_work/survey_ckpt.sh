#!/bin/bash
# 盘点全部历史 run 的 ckpt 与已记录指标曲线(轨迹), 并看 seen20 集结构
cd /root/Workspace/xy/DiT || exit 1

echo "########## A. seen20 集结构 ##########"
for f in assets/eval_top10_seen_20.csv assets/eval_top10_seen_20_slotmap.csv \
         assets/eval_v46_seen20.csv assets/eval_v46_strict200.csv; do
  if [ -f "$f" ]; then
    echo "--- $f : $(( $(wc -l < "$f") - 1 )) 数据行 ---"
    head -2 "$f"
    echo "    槽位分布: $(cut -d, -f13 "$f" 2>/dev/null | tail -n +2 | sort | uniq -c | sort -rn | head -4 | tr '\n' ' ')"
  fi
done

echo
echo "########## B. run 目录 + ckpt 数 + 已有指标文件 ##########"
find assets/results _archive/20261003_twostage -maxdepth 3 -name 'checkpoints' -type d 2>/dev/null \
  | sort | while read -r ck; do
      run=$(dirname "$ck")
      npt=$(ls "$ck"/*.pt 2>/dev/null | wc -l)
      first=$(ls "$ck"/*.pt 2>/dev/null | head -1 | xargs -r basename)
      last=$(ls "$ck"/*.pt 2>/dev/null | tail -1 | xargs -r basename)
      met=$(ls "$run"/*.csv "$run"/*.json 2>/dev/null | xargs -r -n1 basename | tr '\n' ',')
      printf '%-70s ckpt=%-4s %s..%s  files=%s\n' "$run" "$npt" "$first" "$last" "$met"
    done

echo
echo "########## C. 所有 metrics/summary csv ##########"
find assets/results _archive/20261003_twostage exp-std -maxdepth 4 -name '*.csv' 2>/dev/null \
  | grep -viE 'imgs|png' | head -40

echo
echo "########## D. eval_stdskel_summary 与 v46 产物 ##########"
ls -la assets/results/eval_stdskel_summary.csv exp-std/*.csv exp-std/logs/* 2>/dev/null | head -12
tail -5 assets/results/eval_stdskel_summary.csv 2>/dev/null
