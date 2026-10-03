#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== top10_style23 下 skel / shards 目录 ==="
ls -d data/top10_style23/*skel* data/top10_style23/shards_* 2>/dev/null
echo
echo "=== 各候选目录 shard 数 ==="
for d in data/top10_style23/shards_std \
         data/top10_style23/shards_std_w1 \
         data/top10_style23/shards_std_w3 \
         data/top10_style23/shards_std_w7 \
         data/top10_style23/shards_gtskel_w3 \
         data/top10_style23/shards_gtskel_w7; do
  n=$(ls "$d"/*.npz 2>/dev/null | wc -l)
  echo "  $d: $n"
done
echo
echo "=== gt_skel png 档 ==="
ls -d data/top10_style23/gt_skel_png* 2>/dev/null
echo
echo "=== 50k_v2_glyph15k 下的 std 档 ==="
ls -d data/50k_v2_glyph15k/*std* data/50k_v2_glyph15k/*skel* 2>/dev/null
