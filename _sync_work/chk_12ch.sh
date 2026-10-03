#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== docs 里关于 12ch 的结论 ==="
grep -rniE '12ch|12 通道|concat.*(负|放弃|否决|没用)|aux.*(负面|否决)' docs/ 2>/dev/null | head -12
echo
echo "=== v13_base_50k 全曲线 (末 10 行) ==="
tail -10 assets/results/v13_base_50k/eval_stdskel_summary.csv
echo
echo "=== v13_12ch_post 全曲线 (末 10 行) ==="
tail -10 assets/results/v13_12ch_post/eval_stdskel_summary.csv
echo
echo "=== 后续主线(v19+)是否还用 aux/12ch ==="
grep -lE 'aux_latent_shards_dirs|image_channels' src/train/configs/v1[9]*.json src/train/configs/v2*.json 2>/dev/null | head -6
echo "(空 = 后续全部放弃 aux 通道)"
