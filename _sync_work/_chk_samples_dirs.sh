#!/bin/bash
cd /root/Workspace/xy/DiT
for d in s26_ctrl_gt_skel s29_ctrl_gt_skel_1px s31_ctrl_gt_skel_1px ctrl_fame_1pix_v1 s32b_repa_strong s32c_chain; do
  echo "== $d =="
  ls -d assets/results/$d/2026*/checkpoints/eval_samples* 2>/dev/null | sed 's#.*/checkpoints/##'
done
