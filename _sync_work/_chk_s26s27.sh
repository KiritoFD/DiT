#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s26 (GT skel 1px) eval ==="
for f in $(ls assets/results/s26_ctrl_gt_skel/*/checkpoints/eval_auto_*.json 2>/dev/null | sort); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"  step {d.get('step',0):>5}  ssim={d.get('ssim',0):.4f}  lpips={d.get('lpips',0):.4f}  skel_iou={d.get('skel_iou',0):.4f}\")" 2>/dev/null
done | tail -8
echo ""
echo "=== s27 (std skel) eval ==="
for f in $(ls assets/results/s27_ctrl_data/skel/std_skel/*/checkpoints/eval_auto_*.json 2>/dev/null | sort); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"  step {d.get('step',0):>5}  ssim={d.get('ssim',0):.4f}  lpips={d.get('lpips',0):.4f}  skel_iou={d.get('skel_iou',0):.4f}\")" 2>/dev/null
done | tail -8
echo ""
echo "=== 1px skel latent dirs ==="
ls -d data/skel/final_skel_latents_fame_1px data/skel/final_skel_latents_eval_1px data/skel/final_skel1_fame data/skel/std_skel1_latents_fame data/skel/std_skel1_latents_eval 2>/dev/null
echo ""
echo "=== ctrl_fame_1pix_v1 / 1pix results ==="
ls assets/results/ | grep -a -i '1pix\|1px'
