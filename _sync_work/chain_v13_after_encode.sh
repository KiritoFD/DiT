#!/bin/bash
# 链式: 等 inst-skel encode 结束 -> 拉起 v13 (XS/2, E1)
set -u
cd /root/Workspace/xy/DiT
while pgrep -f build_skel_latents > /dev/null; do sleep 30; done
sleep 10
N=$(ls data/aux/inst_skel_latents_px60/shard_*.npz 2>/dev/null | wc -l)
echo "[chain] encode done, shards=$N"
if [ "$N" -gt 0 ]; then
    bash _sync_work/_launch_v12_S_cat.sh v13 v13_pretrain_XS_cat_fame_kxl_tj_px60 v13_pretrain_XS_cat_fame_kxl_tj_px60
else
    echo "[chain] no shards found -> NOT launching"
fi
