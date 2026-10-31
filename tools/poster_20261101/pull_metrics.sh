#!/bin/bash
# Collect the official eval summary/metrics files for each poster column.
set -u
cd /root/Workspace/xy/DiT || exit 1
OUT=/tmp/poster_metrics; rm -rf "$OUT"; mkdir -p "$OUT"

take() {  # take <name> <find-root> <extra-find-args...>
  local name=$1 root=$2; shift 2
  local n=0
  while read -r f; do
    [ -z "$f" ] && continue
    mkdir -p "$OUT/$name/$(dirname "${f#./}")"
    cp -p "$f" "$OUT/$name/${f#./}" && n=$((n+1))
  done < <(find "$root" "$@" 2>/dev/null | head -12)
  echo "  $name: $n files"
}

echo "=== summary/metrics files per column ==="
take S9_50k  assets/style_matrix_20261028/generation2full/4090/runs/S9 \
  \( -name 'eval_stdskel_summary.csv' -o -name 'eval_auto*.json' -o -name '*_metrics.json' \) -size -400k
take S0_85k  assets/style_focus_85k_20261029/4090/runs/S0 \
  \( -name 'eval_stdskel_summary.csv' -o -name 'eval_auto*.json' -o -name '*_metrics.json' \) -size -400k
take A_35k   assets/abc_20261031/execution/runs/A \
  \( -name 'eval_stdskel_summary.csv' -o -name 'eval_auto*.json' -o -name '*_metrics.json' \) -size -400k
take P3A_20k assets/p125_20261030/execution/runs/P3A \
  \( -name 'eval_stdskel_summary.csv' -o -name 'eval_auto*.json' -o -name '*_metrics.json' \) -size -400k
take A_43429 assets/abc_20261031/execution/runs/A \
  \( -name 'eval_stdskel_summary.csv' -o -name 'eval_auto*.json' -o -name '*_metrics.json' \) -size -400k

echo
echo "=== inventory ==="
find "$OUT" -type f | sed "s|$OUT/||" | head -40
du -sh "$OUT"
cd /tmp && tar czf poster_metrics.tar.gz poster_metrics && ls -l poster_metrics.tar.gz
