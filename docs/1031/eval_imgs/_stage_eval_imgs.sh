#!/bin/bash
# Stage a curated, repo-sized subset of historical eval images.
# Rules: all posters/grids/summaries; all small eval metadata trees; for each
# eval panel directory keep the first N generated + N GT images (deterministic).
cd /root/Workspace/xy/DiT || exit 1
STAGE=/tmp/evalimgs
rm -rf "$STAGE"; mkdir -p "$STAGE"
N=6

copy_keep() {   # copy_keep <src> <dest-root>
  local src=$1 root=$2
  [ -f "$src" ] || return 0
  local rel=${src#./}
  mkdir -p "$root/$(dirname "$rel")"
  cp -p "$src" "$root/$rel"
}

echo "[1/4] posters, grids, summaries, panels"
find assets docs -type f \( -name 'poster*.png' -o -name '*summary*.png' \
  -o -name '*grid*.png' -o -name '*panel*.png' \) 2>/dev/null \
  | while read -r f; do copy_keep "$f" "$STAGE"; done
find assets/posters -type f 2>/dev/null | while read -r f; do copy_keep "$f" "$STAGE"; done

echo "[2/4] small eval metadata trees"
for d in assets/eval_hwd_20261029 assets/full_eval_raw assets/eval_strata; do
  [ -d "$d" ] || continue
  mkdir -p "$STAGE/$(dirname "$d")"
  cp -a "$d" "$STAGE/$d"
done

echo "[3/4] eval panel samples (first $N generated + $N GT per panel dir)"
find assets docs -type d -name eval_samples_ctrl 2>/dev/null | while read -r d; do
  find "$d" -type d -mindepth 1 -maxdepth 2 | while read -r sub; do
    ls "$sub" 2>/dev/null | grep -E '^g[0-9]+\.png$' | sort -V | head -n "$N" \
      | while read -r f; do copy_keep "$sub/$f" "$STAGE"; done
    ls "$sub" 2>/dev/null | grep -E '^gt[0-9]+\.png$' | sort -V | head -n "$N" \
      | while read -r f; do copy_keep "$sub/$f" "$STAGE"; done
  done
done

echo "[4/4] ink_eval samples (first $N per subdir)"
find assets/ink_eval -type d -mindepth 1 2>/dev/null | while read -r sub; do
  ls "$sub" 2>/dev/null | grep -E '\.png$' | sort | head -n "$N" \
    | while read -r f; do copy_keep "$sub/$f" "$STAGE"; done
done

echo
echo "=== staged ==="
find "$STAGE" -name '*.png' | wc -l | sed 's/^/png files: /'
du -sh "$STAGE"
du -sh "$STAGE"/* 2>/dev/null | sort -hr | head -12
