#!/usr/bin/env bash
# 关键: eval 到底跑了没有? (配置注释警告过 --in-mem-eval 会静默失效)
set -u
cd /root/Workspace/xy/DiT || exit 1
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
RUN=$(cat exp-std/runs_AB/_active_ckpt_dir.txt 2>/dev/null | tail -1)
echo "log=$LOG"
echo "active_ckpt_dir=$RUN"
echo
echo "=== [1] 日志里有没有 eval 痕迹 ==="
grep -anE 'in-mem-eval|in_mem_eval|eval200|ssim|strict_pred|\[eval\]|poster|GPU eval' "$LOG" 2>/dev/null | tail -20
echo "(以上为空 = eval 没跑)"
echo
echo "=== [2] run 目录里有没有 eval 产物 ==="
D=$(ls -dt exp-std/runs_AB/*/ 2>/dev/null | head -1)
echo "run dir = $D"
ls -la "$D" 2>/dev/null | head -20
echo "--- eval csv / 采样目录 ---"
find "$D" -maxdepth 2 -name '*eval*' -o -maxdepth 2 -name 'eval_samples_ctrl' 2>/dev/null | head -10
echo "--- ckpt ---"
ls -la "$D"/checkpoints/ 2>/dev/null | tail -6
echo
echo "=== [3] 配置的 eval 开关 ==="
grep -aE 'in_mem_eval|eval_cfg|eval_steps|eval_mode|ckpt_every' src/train/configs/v50_A_space_xattn_style_adaLN_top10.json
echo
echo "=== [4] 时间戳连续性: 跨 5000 步落点有没有停顿 ==="
grep -aE 'step=000' "$LOG" | awk '{print $1, $2}' | sed 's/\x1b\[[0-9;]*m//g' | tail -30
