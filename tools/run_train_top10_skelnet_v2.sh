#!/usr/bin/env bash
# tools/run_train_top10_skelnet_v2.sh — 在 4090 上针对 11% 闭合率短板，快训 SkelNet-V2 拓扑分支 (剪刀与胶水)
set -e
cd /root/Workspace/xy/DiT

echo "=== 开始训练 Top 10 SkelNet-V2 (剪刀与胶水拓扑增强版) ==="
echo "输入数据: assets/train_top10_style23.csv (38,583 样本)"
echo "风格表: assets/callig_script_emb_top10.pt (23 槽位)"
echo "热启动底座: assets/deform_skel_top10_v1.pt"
echo "目标产出: assets/deform_skel_top10_v2.pt"

/opt/conda/envs/cu121/bin/python tools/train_deform_standalone.py \
    --csv assets/train_top10_style23.csv \
    --std-dir data/top10_style23/shards_std \
    --gt-dir data/top10_style23/shards_aux_skel3 \
    --style-emb assets/callig_script_emb_top10.pt \
    --init assets/deform_skel_top10_v1.pt \
    --out assets/deform_skel_top10_v2.pt \
    --topo-mode 1 \
    --w-prune-l1 0.02 \
    --w-lig-l1 0.02 \
    --freeze-base 1 \
    --steps 3000 \
    --batch 2048 \
    --group 32 \
    --lr 5e-4 \
    --w-contr 0.5 \
    --contr-mode margin \
    --contr-margin 0.008 \
    --contr-hard 4 \
    --stroke-mod 1 \
    --width 96 \
    --save-every 500 \
    --eval-every 250
