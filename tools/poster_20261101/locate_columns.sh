#!/bin/bash
# Locate a usable eval panel dir per poster column, on this host.
cd /root/Workspace/xy/DiT || exit 1
probe() {  # probe <label> <find-expr...>
  local label=$1; shift
  local hits; hits=$(find "$@" 2>/dev/null | head -4)
  if [ -z "$hits" ]; then echo "MISSING  $label"; else
    echo "HIT  $label"
    echo "$hits" | while read -r d; do
      echo "      g=$(ls "$d"/g*.png 2>/dev/null | grep -cE '/g[0-9]+\.png$')  $d"
    done
  fi
}

echo "=== A (abc chain) evaluations ==="
ls -d assets/abc_20261031/execution/runs/A/evaluations/*/ 2>/dev/null | tail -6
echo
echo "=== P3A evaluations ==="
ls -d assets/p125_20261030/execution/runs/P3A/evaluations/*/ 2>/dev/null | tail -6
echo
echo "=== 43429 anywhere ==="
find assets/abc_20261031 -type d -name '*43429*' 2>/dev/null | head -6
echo
probe "Top10 70k"   assets/results -maxdepth 6 -type d -name eval200fix -path '*armA_writer_decoupled_70k*'
probe "DiT L20 200k" assets/results scratch -maxdepth 9 -type d -name eval200fix -path '*v90A_20L*'
probe "S0 85k"      assets/style_focus_85k_20261029 -type d -name eval200fix -path '*0085000*'
probe "S9 50k"      assets/style_matrix_20261028 -type d -name eval200fix -path '*0050000*'
probe "v80 200k"    assets/results -maxdepth 6 -type d -name eval200fix -path '*v80*step0200000*'
