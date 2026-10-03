#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== ctrl_fame_1pix_v1 dir ==="
find assets/results/ctrl_fame_1pix_v1 -maxdepth 2 | head -15
echo ""
echo "=== 相关配置文件 ==="
find . -maxdepth 3 -name '*.json' | grep -a -i -E '1pix|1px|ctrl' | head -10
echo ""
echo "=== s26 train log 位置 ==="
find assets/results/s26_ctrl_gt_skel -name '*.log' -o -name 'log.txt' 2>/dev/null | head -5
echo ""
echo "=== s26 tmux log ==="
ls -la assets/results/s26_ctrl_gt_skel/*.log 2>/dev/null
echo "=== s26 train.log 内容前 5 行 ==="
head -8 assets/results/s26_ctrl_gt_skel/train.log 2>/dev/null || echo "(no train.log)"
