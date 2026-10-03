#!/bin/bash
cd /root/Workspace/xy/DiT
for spec in \
  "s26:assets/results/s26_ctrl_gt_skel/20260831-122923-s26-ctrl-gt-skel-1pix/checkpoints/eval_samples_ctrl/step0085000" \
  "s31:assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/eval_samples_ctrl/step0030000" \
  "1pix:assets/results/ctrl_fame_1pix_v1/20260830-205652-fame-ctrl-skel-1pix-v1/checkpoints/eval_samples_ctrl/step0030000" \
  "s32b:assets/results/s32b_repa_strong/20260901-204250-s32b-repa-strong/checkpoints/eval_samples_ctrl/step0020000" \
  "s32c:assets/results/s32c_chain/20260902-004653-s32c-repa-longconv/checkpoints/eval_samples_ctrl/step0030000" ; do
  name="${spec%%:*}"; dir="${spec#*:}"
  echo "== $name =="
  ls "$dir" 2>/dev/null | head -6
  echo "  (gt/sample 计数):"
  ls "$dir" 2>/dev/null | grep -c "^gt"
  ls "$dir" 2>/dev/null | grep -c "^sample"
  if [ -d "$dir/ctrl" ]; then echo "  -> has ctrl/ subdir:"; ls "$dir/ctrl" 2>/dev/null | head -3; fi
  if [ -d "$dir/base" ]; then echo "  -> has base/ subdir:"; ls "$dir/base" 2>/dev/null | head -3; fi
done
