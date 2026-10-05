#!/usr/bin/env bash
cd /root/Workspace/xy/DiT || exit 1
L(){ echo; echo "########## $* ##########"; }

L "1 dit.py 281-300 / 760-800 (两个 xattn 实现)"
sed -n '281,300p' src/model/dit.py
echo "..."
sed -n '760,800p' src/model/dit.py

L "2 dit.py 1700-1745 (mode 选择)"
sed -n '1700,1745p' src/model/dit.py

L "3 cli.py 所有 add_argument"
grep -n 'add_argument' src/train/cli.py | sed 's/^\s*//' | head -60

L "4 骨架损失/skel_head/probe 全仓库定位"
grep -rn 'LatentSkelStructureLoss\|w_latent_skel\|skel_head\|latent_skel_probe' --include=*.py src tools | head -25

L "5 预训练加载/冻结相关开关"
grep -rn "add_argument(\"--\(init\|pretrained\|base-ckpt\|from-ckpt\|freeze\)" src/train/cli.py | head -25
echo "--- freeze policy 分支条件 ---"
sed -n '1043,1060p;1117,1135p;1180,1200p;1255,1275p' src/train/train.py

L "6 v54 config 余下部分 (batch/eval)"
sed -n '60,120p' src/train/configs/v54_minimal_tables_noskel.json

L "7 v65 现状"
V65=$(ls -dt assets/results/v65_skel_inj2468/*/ 2>/dev/null | head -1); V65=${V65%/}
grep -ao 'step=[0-9]*' "$V65/log.txt" | tail -1
grep -a 'Steps/Sec' "$V65/log.txt" | tail -1 | sed 's/\x1b\[[0-9;]*m//g'
