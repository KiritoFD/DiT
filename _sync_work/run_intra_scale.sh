#!/bin/bash
# 大规模 intra (输出多样性) 评测: cfg 0.7 与 1.0 各一份
#   30 个条件 x 4 个种子 = 120 张/份, intra 对数 30*C(4,2)=180 对
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V12=assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60/20260916-182657-v12-pretrain-S-cat-fame-kxl-tj-px60/checkpoints/0100000.pt
EC=assets/eval_fame3_strict_clean_v9.csv
SK=data/fame-kxl-tj-px60/shards_std_eval

for CFG in 0.7 1.0; do
    T=$(echo "$CFG" | tr -d '.')
    rm -f /tmp/intra$T.log
    setsid nohup env LC_ALL=C.UTF-8 $PY -u tools/eval_diversity.py \
        --ckpt $V12 --eval-csv $EC --n-cond 30 --k 4 --cfg $CFG \
        --device cpu --threads 28 --skel-shards $SK \
        --out-tag v12_intra_cfg$T > /tmp/intra$T.log 2>&1 < /dev/null &
done
echo launched
