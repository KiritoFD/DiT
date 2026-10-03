#!/bin/bash
# v18 (w_style_rank=0.005, 200k) 的实际结果 —— 这是"端点风格损失有没有用"的直接证据
cd /root/Workspace/xy/DiT || exit 1
D=assets/results/v18_style_rank_200k

echo "########## 1. v18 目录 ##########"
ls -la "$D"/ 2>/dev/null | head -14
echo "--- ckpt ---"
ls -t "$D"/*/checkpoints/*.pt 2>/dev/null | head -4

echo
echo "########## 2. v18 配置里的关键项 ##########"
grep -E '"(w_style_rank|style_rank_ckpt|style_rank_t_min|style_rank_t_max|w_repa|w_std_mid|max_steps|lr|data_csv)"' \
     src/train/configs/v18_style_rank_200k.json 2>/dev/null

echo
echo "########## 3. v18 全部评测行 (summary + 备份, 表头: exp,step,set,n,ssim...) ##########"
for f in "$D"/eval_stdskel_summary.csv "$D"/eval_stdskel_summary.csv.bak_oldcols*; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f")"
  head -1 "$f"
  tail -n +2 "$f" | head -12
done 2>/dev/null

echo
echo "########## 4. v18 训练日志 (最后一个) ##########"
L=$(ls -t logs/v18_series/*.log 2>/dev/null | head -1)
echo "L=$L"
if [ -n "$L" ]; then
  echo "--- 头 6 行(启动) ---"; head -6 "$L"
  echo "--- 尾 12 行 ---"; tail -12 "$L"
  echo "--- 所有 in-mem-eval 行 ---"
  grep -E 'in-mem-eval.*ssim=' "$L" | tail -12
fi

echo
echo "########## 5. 对照: v17 基模同口径 (文档记录 0.5454 @100k) ##########"
ls -d assets/results/v17* 2>/dev/null | head -4
