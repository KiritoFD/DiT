#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s26 dir structure ==="
ls assets/results/s26_ctrl_gt_skel/
echo "=== s26 latest run ==="
L=$(ls -dt assets/results/s26_ctrl_gt_skel/2026*/ 2>/dev/null | head -1)
echo "run: $L"
ls "$L"checkpoints/ 2>/dev/null | tail -10
echo "=== s26 eval files (non eval_auto) ==="
find assets/results/s26_ctrl_gt_skel -name '*.json' -path '*eval*' 2>/dev/null | tail -10
echo "=== s26 ctrl eval json sample ==="
f=$(find assets/results/s26_ctrl_gt_skel -name '*eval*.json' | tail -1)
if [ -n "$f" ]; then echo "file: $f"; /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print({k:round(v,4) if isinstance(v,float) else v for k,v in d.items() if k in ('step','ssim','lpips','skel_iou','mse','ctrl_ssim','ctrl_skel_iou')})" 2>/dev/null; fi
echo ""
echo "=== s27 latest run ==="
ls -dt assets/results/s27_ctrl_data/skel/std_skel/2026*/ 2>/dev/null | head -2
echo "=== s27 eval files ==="
find assets/results/s27_ctrl_data/skel/std_skel -name '*.json' -path '*eval*' 2>/dev/null | tail -6
echo ""
echo "=== ctrl_fame_1pix_v1 (早期 1px GT) eval ==="
find assets/results/ctrl_fame_1pix_v1 -name '*eval*.json' 2>/dev/null | tail -5
for f in $(find assets/results/ctrl_fame_1pix_v1 -name '*eval*.json' 2>/dev/null | sort | tail -3); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"  step {d.get('step',0):>5}  ssim={d.get('ssim',0):.4f}  skel_iou={d.get('skel_iou',0):.4f}  lpips={d.get('lpips',0):.4f}\")" 2>/dev/null
done
