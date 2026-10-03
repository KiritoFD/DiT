#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s26 latest run ctrl eval (last 6) ==="
for f in $(ls assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1px/checkpoints/eval_auto_ctrl_*.json | sort | tail -6); do
  echo "--- $f ---"
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(json.dumps(d, indent=0)[:500])"
done
echo ""
echo "=== s26 train.log eval lines ==="
grep -a -E 'ctrl.*(ssim|skel_iou)|step.*ssim|ctrl-eval|eval.*ctrl' assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1px/log.txt 2>/dev/null | tail -10
echo ""
echo "=== s27 directory ==="
ls -la assets/results/s27_ctrl_data/skel/std_skel/ 2>/dev/null || echo "(s27 dir missing)"
