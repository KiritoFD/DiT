#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
check() {  # check <label> <glob-dir>
  local label=$1 pat=$2 d
  d=$(ls -d $pat 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "MISSING  $label  ($pat)"; return; fi
  local n; n=$(ls "$d"/g*.png 2>/dev/null | grep -cE '/g[0-9]+\.png$')
  local gt; gt=$(ls "$d"/gt*.png 2>/dev/null | grep -cE '/gt[0-9]+\.png$')
  echo "OK  g=$n gt=$gt  $label  -> $d"
}

echo "=== eval200fix family (row axis: 187-row panel) ==="
check "Top2 30k"        "assets/results/unet_subsets/u100_base_top2_noaug/*/eval_samples_ctrl/step0030000/eval200fix"
check "Top5 Scale 40k"  "assets/results/unet_subsets/top5_u100_deep_sym4_50k/*/eval_samples_ctrl/step0040000/eval200fix"
check "Top10 70k"       "assets/results/unet_decoupled_114m/armA_writer_decoupled_70k/*/eval_samples_ctrl/step0070000/eval200fix"
check "DiT L20 200k"    "scratch/v90_20l_gpu_20261017_01/assets/results/v90A_20L_stdstrong/*/eval_samples_ctrl/step0200000/eval200fix"

echo
echo "=== val100 family (row axis: 100-row panel) ==="
check "P3A 20k"  "assets/p125_20261030/execution/runs/P3A/eval_samples_ctrl/step0020000/eval200fix"
check "A 43429"  "assets/abc_20261031/execution/runs/A/eval_samples_ctrl/step0043429/eval200fix"

echo
echo "=== what eval dirs does A actually have? ==="
ls -d assets/abc_20261031/execution/runs/A/eval_samples_ctrl/*/ 2>/dev/null | tail -5
ls assets/abc_20261031/execution/runs/A/eval_samples_ctrl/*/eval200fix/ 2>/dev/null | head -3
echo
echo "=== P3A ==="
ls -d assets/p125_20261030/execution/runs/P3A/eval_samples_ctrl/*/ 2>/dev/null | tail -3
