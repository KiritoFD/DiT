#!/usr/bin/env bash
# tools/run_train_top10_skelnet.sh — 在 4090 上专精训练 23 槽位 Top 10 SkelNet 几何形变网络
set -e
cd /root/Workspace/xy/DiT

echo "=== 开始训练 Top 10 (23 槽位) 专精 SkelNet 几何形变网络 ==="
echo "输入数据: assets/train_top10_style23.csv (38,583 样本)"
echo "风格表: assets/callig_script_emb_top10.pt (23 槽位)"
echo "热启动权重: assets/deform_skel_v10.pt"
echo "目标产出: assets/deform_skel_top10_v1.pt"

/opt/conda/envs/cu121/bin/python tools/train_deform_standalone.py \
    --csv assets/train_top10_style23.csv \
    --std-dir data/top10_style23/shards_std \
    --gt-dir data/top10_style23/shards_aux_skel3 \
    --style-emb assets/callig_script_emb_top10.pt \
    --init assets/deform_skel_v10.pt \
    --out assets/deform_skel_top10_v1.pt \
    --steps 8000 \
    --batch 2048 \
    --group 32 \
    --lr 1e-3 \
    --w-contr 0.5 \
    --contr-mode margin \
    --contr-margin 0.008 \
    --contr-hard 4 \
    --stroke-mod 1 \
    --width 96 \
    --save-every 1000 \
    --eval-every 500
