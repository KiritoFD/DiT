#!/bin/bash
# 端到端联合训练 (v34_e2e) 的下游判分: 用它自己配对的冻结主干 v32 评判,
# 逐个 step 落点 (1000..5000) 打 seen20 / strict84 的 ssim。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V32=$(ls -d assets/results/v32_stage2_img/*/checkpoints/0080000.pt 2>/dev/null | head -1)
echo "backbone = $V32"
for s in 001000 002000 003000 004000 005000; do
  echo "===== step $s ====="
  $PY -u tools/run_skel_calibration.py --ckpt "$V32" --alphas 0 \
    --pred-seen   "assets/results/v34_e2e/predskel_step$s/seen20" \
    --pred-strict "assets/results/v34_e2e/predskel_step$s/strict84" \
    --out "assets/results/_calib_v34e2e_$s" 2>&1 | grep -iE 'strict|seen|ssim|pred' | tail -8
done
echo "===== V34_E2E_EVAL_DONE ====="
