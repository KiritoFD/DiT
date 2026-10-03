#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== gt_skel png 目录 ==="
ls -d data/*/gt_skel* data/*/*gt_skel* 2>/dev/null
echo
echo "=== 骨架相关 shard 目录 (取样) ==="
ls -d data/*/*skel* 2>/dev/null | head -30
echo
echo "=== 训练 csv 规模 ==="
for f in assets/train_top10_style23_minusval.csv assets/train_base_sym_clean.csv assets/train_fame-kxl-tj-px60.csv assets/train_top10_style23.csv assets/val_skelnet.csv; do
  if [ -f "$f" ]; then printf "%-52s %s\n" "$f" "$(wc -l < $f)"; fi
done
echo
echo "=== gt_skel_png 数量 ==="
for d in data/top10_style23/gt_skel_png data/50k_v2_glyph15k/gt_skel_png data/top10_style23/gt_skel_w7_png; do
  printf "%-52s %s\n" "$d" "$(ls $d 2>/dev/null | wc -l)"
done
echo
echo "=== top10_style23 下所有数据目录 ==="
ls -d data/top10_style23/*/ 2>/dev/null
echo
echo "=== 50k_v2_glyph15k 下所有数据目录 ==="
ls -d data/50k_v2_glyph15k/*/ 2>/dev/null
