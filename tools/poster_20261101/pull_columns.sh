#!/bin/bash
# Copy exactly the poster's 21 cells from each remote column into /tmp/poster_imgs.
set -u
cd /root/Workspace/xy/DiT || exit 1
OUT=/tmp/poster_imgs
rm -rf "$OUT"; mkdir -p "$OUT"

IDX200="147 171 159 161 164 30 41 24 45 85 88 91 94 124 121 127 1 4 97 101 145"
IDX100="48 46 43 44 49 79 74 78 71 9 6 2 8 16 15 11 94 99 65 62 58"

grab200() {  # grab200 <name> <panel-dir>
  local name=$1 d=$2 n=0 miss=0
  mkdir -p "$OUT/$name"
  for i in $IDX200; do
    if [ -f "$d/g$i.png" ]; then cp -p "$d/g$i.png" "$OUT/$name/g$i.png"; n=$((n+1));
    else echo "  MISS $name g$i.png"; miss=$((miss+1)); fi
  done
  echo "  $name: $n copied, $miss missing"
}
grab100() {  # grab100 <name> <panel-dir>
  local name=$1 d=$2 n=0 miss=0
  mkdir -p "$OUT/$name"
  for i in $IDX100; do
    if [ -f "$d/g$i.png" ]; then cp -p "$d/g$i.png" "$OUT/$name/g$i.png"; n=$((n+1));
    else echo "  MISS $name g$i.png"; miss=$((miss+1)); fi
  done
  echo "  $name: $n copied, $miss missing"
}

echo "=== 187-row (eval200fix) columns ==="
grab200 S9_50k   assets/style_matrix_20261028/generation2full/4090/runs/S9/evaluations/0050000/heldout187_seed0/eval_samples_ctrl/step0050000/eval200fix
grab200 S0_85k   assets/style_focus_85k_20261029/4090/runs/S0/evaluations/0085000/heldout187_seed0/eval_samples_ctrl/step0085000/eval200fix
grab200 A_35k    assets/abc_20261031/execution/runs/A/evaluations/0035000/heldout187_seed0/eval_samples_ctrl/step0035000/eval200fix

echo "=== 100-row (val100) columns ==="
grab100 P3A_20k  assets/f_20261101/eval/K_P3A/eval_samples_ctrl/step0020000/eval200fix
grab100 A_43429  assets/f_20261101/eval/K_A/eval_samples_ctrl/step0043429/eval200fix

echo
echo "=== totals ==="
for d in "$OUT"/*; do echo "  $(ls "$d" | wc -l)  $(basename "$d")"; done
du -sh "$OUT"
cd /tmp && tar czf poster_imgs.tar.gz poster_imgs && ls -l poster_imgs.tar.gz
