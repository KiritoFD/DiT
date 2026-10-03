#!/bin/bash
cd /root/Workspace/xy/DiT
for d in v8_3stage s32c_chain s32b_repa_strong s32_repa_finetune s31_ctrl_gt_skel_1px s30_dino_char_strong_pretrain s29_ctrl_gt_skel_1px; do
  echo "=== $d ==="
  RUN=$(ls -dt assets/results/$d/2026*/ 2>/dev/null | head -1)
  if [ -z "$RUN" ]; then echo "  (no run)"; continue; fi
  echo "  run: $RUN"
  echo "  eval_auto: $(ls "$RUN"checkpoints/eval_auto_*.json 2>/dev/null | wc -l)"
  echo "  eval_auto_ctrl: $(ls "$RUN"checkpoints/eval_auto_ctrl_*.json 2>/dev/null | wc -l)"
  # 显示最新 eval 的字段
  F=$(ls "$RUN"checkpoints/eval_auto_*.json "$RUN"checkpoints/eval_auto_ctrl_*.json 2>/dev/null | sort | tail -1)
  if [ -n "$F" ]; then
    /opt/conda/bin/python -c "
import json
d=json.load(open('$F'))
print('  file:', '$F'.split('/')[-1])
if 'ctrl' in d and isinstance(d.get('ctrl'), dict):
    print('  keys:', list(d.keys()))
    for br in ('base','ctrl'):
        if br in d:
            v=d[br]
            print(f'    {br}: step={d.get(\"step\")} ssim={v.get(\"ssim\",0):.4f} lpips={v.get(\"lpips\",0):.4f} skel_iou={v.get(\"skel_iou\",0):.4f}')
else:
    print(f'  step={d.get(\"step\")} ssim={d.get(\"ssim\",0):.4f} lpips={d.get(\"lpips\",0):.4f} skel_iou={d.get(\"skel_iou\",0):.4f}')
"
  fi
  echo ""
done
