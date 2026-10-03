#!/usr/bin/env bash
# 判定: 这些审计项会不会让"训练本身"无效?
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "======== [1] 曲线是否还能重建 (eval_auto_*.json 在不在) ========"
find exp-std/runs_AB -name 'eval_auto_*.json' 2>/dev/null | head -10
echo "  数量: $(find exp-std/runs_AB -name 'eval_auto_*.json' 2>/dev/null | wc -l)"
D=$(sed -n '1p' exp-std/runs_AB/_active_ckpt_dir.txt 2>/dev/null | xargs dirname 2>/dev/null)
echo "  当前 run 目录: $D"
ls -la "$D"/eval_auto_*.json 2>/dev/null | tail -5
echo "  --- 内容样例 ---"
for f in $(find exp-std/runs_AB -name 'eval_auto_*.json' 2>/dev/null | sort | tail -2); do
  echo "  $f:"; head -c 400 "$f"; echo
done

echo
echo "======== [2] np.empty 垃圾路径是否被触发 (skel 变体缺 id 告警) ========"
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
grep -a '找不到\|⚠' "$A" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -10 || echo "  ✓ 无任何缺失告警 (np.empty 垃圾路径未被触发)"
echo "  --- 其余告警扫描 ---"
grep -a 'warn\|⚠' "$A" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -5

echo
echo "======== [3] 事实核对: 备份文件 vs 主文件 ========"
ls -la exp-std/runs_AB/eval_stdskel_summary.csv 2>/dev/null || echo "  ✗ 主 summary 不存在 (印证 P0-1)"
ls -la exp-std/runs_AB/eval_stdskel_summary.csv.bak_oldcols* 2>/dev/null | tail -4
echo "  --- 备份文件里有什么 ---"
for f in exp-std/runs_AB/eval_stdskel_summary.csv.bak_oldcols*; do
  [ -f "$f" ] || continue
  echo "  $(basename $f): $(wc -l < "$f") 行"
  head -2 "$f" | cut -c1-120
done

echo
echo "======== [4] 当前训练是否健康 ========"
grep -a 'Steps/Sec' "$A" 2>/dev/null | tail -2 | sed 's/\x1b\[[0-9;]*m//g'
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
date
