#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s26 train.log full (first 60 lines) ==="
head -60 assets/results/s26_ctrl_gt_skel/train.log 2>/dev/null
echo ""
echo "=== s26_tmux.log 关键行 ==="
grep -a -E 'Traceback|Error|error|skel|Skel|ctrl|Ctrl|Dataset|latent' assets/results/s26_ctrl_gt_skel/s26_tmux.log 2>/dev/null | head -20
