#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== 当前 stage1 状态 ==="
tail -6 /tmp/stage1_bb.log 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo
echo "=== v13 系列 config 全文 ==="
for f in src/train/configs/v13_base_50k.json src/train/configs/v13_12ch_post.json \
         src/train/configs/v13_styletok.json; do
  echo "--- $f"
  cat "$f" 2>/dev/null | head -40
  echo
done
echo "=== 所有 config 里 latent_shards_dir 指向骨架(非 img)的 ==="
grep -l '"latent_shards_dir".*skel' src/train/configs/*.json 2>/dev/null | head -12
echo
echo "=== 这些 config 的目标 ==="
for f in $(grep -l '"latent_shards_dir".*skel' src/train/configs/*.json 2>/dev/null | head -12); do
  echo "  $f: $(grep -oE '"latent_shards_dir":[^,]*' $f | head -1) $(grep -oE '"skel_latent_shards_dir":[^,]*' $f | head -1)"
done
