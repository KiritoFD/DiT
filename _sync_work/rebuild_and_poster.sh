#!/bin/bash
# 重建数据集为 0/255 -> 出取样 poster -> 转 jpg
cd /root/Workspace/xy/DiT
pkill -f 'train_skelnet_[fp]m' 2>/dev/null
sleep 4
echo "===== 重建数据集 (0/255) ====="
/opt/conda/envs/cu121/bin/python -u tools/build_skel64_dataset.py --res 64 2>&1 \
    | grep -E '落盘|墨占比|骨架为空|DONE'
echo "===== 数据集取样 poster ====="
/opt/conda/envs/cu121/bin/python -u tools/poster_skel64_dataset.py --split train --n 8 2>&1 | tail -4
/opt/conda/envs/cu121/bin/python -u tools/poster_to_jpg.py 1100 _ot_scratch/skel64_dataset.png /tmp/ds64.jpg 2>&1 | tail -1
