#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s21 step0030000 (base) ==="
ls assets/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/eval_samples/step0030000/ | head -5
ls assets/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/eval_samples/step0030000/ | wc -l
echo "=== s32c step0030000 (repa) ==="
ls assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/eval_samples_ctrl/step0030000/ | head -5
ls assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/eval_samples_ctrl/step0030000/ | wc -l
echo "=== s30 step0030000 (base) ==="
ls assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/eval_samples/step0030000/ | head -5
