#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s21_fame_flow_v2 eval metrics ==="
for f in $(ls assets/results/s21_fame_flow_v2/*/checkpoints/eval_auto_*.json | sort); do
  /opt/conda/bin/python -c "import json;d=json.load(open('$f'));print(f\"step {d['step']:>6}  ssim={d['ssim']:.4f}  lpips={d['lpips']:.4f}  mse={d['mse']:.4f}  skel_iou={d['skel_iou']:.4f}\")"
done
echo ""
echo "=== s21 config key fields ==="
/opt/conda/bin/python -c "
import json
d=json.load(open('assets/results/s21_fame_flow_v2/*/resolved_config.json'))
for k in ['model','char_embed_dim','freeze_char_table','num_characters','diffusion_type','eval_cfg','eval_steps']:
    print(f'  {k}: {d.get(k)}')" 2>/dev/null || echo "(no resolved config)"
echo "=== s21 config file ==="
ls src/train/configs/s21_fame_flow_v2.json && grep -a -E 'char_embed|freeze_char|model|eval_cfg|eval_steps' src/train/configs/s21_fame_flow_v2.json | head
