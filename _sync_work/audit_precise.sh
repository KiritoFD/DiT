#!/usr/bin/env bash
# 复查上一轮的子串假阳性 (精确条件)
set -u
cd /root/Workspace/xy/DiT || exit 1
V() { printf '%-52s %s\n' "$1" "$2"; }

echo "======== 精确复查 ========"

echo "--- [1] '白纸/空白' 基线: 是否真的算过 (上一轮是假阳性) ---"
grep -n '白纸\|blank_baseline\|empty_baseline\|zeros_like(gt\|np.zeros_like(gt' src/eval/in_mem_eval.py | head -5 \
  || echo "  未找到 -> P0-2c 不成立"
echo "  实际基线对照项 (从 eval 行拆解):"
grep -a 'in-mem-eval. step 15000 done' exp-std/logs_AB/A_*.log 2>/dev/null | tail -1 | cut -c1-200

echo
echo "--- [2] std shard x eval csv 全量 id 核对工具: 真实文件名 ---"
ls tools/ | grep -iE 'check|verify|audit|coverage|cross|alignment' | head -10
echo "  (若上面只有我们本轮新写的 check_style_asset/audit_*, 则该项不成立)"

echo
echo "--- [3] fresh_scheduler 默认值 ---"
grep -n -B1 -A3 'fresh-scheduler' src/train/cli.py | head -12

echo
echo "--- [4] _pred 目录缺失时的行为 ---"
grep -n '_pred' src/eval/in_mem_eval.py | head -10
echo "  --- 有无 'missing' / 'return None' / GT 兜底 ---"
grep -n -A3 'pred_dir\|_pred =' src/eval/in_mem_eval.py | head -20

echo
echo "--- [5] 选模指标: 早停/选模到底看什么 ---"
grep -n 'early_stop_metric' src/train/cli.py | head -3
grep -n 'metric=iou_lpips' src/train/early_stop.py src/train/train.py | head -3
echo "  --- iou_lpips 的组合方式 ---"
grep -n -A6 'iou_lpips' src/train/early_stop.py | head -20
