#!/bin/bash
# ★ 门槛测试: 只拟合 N 条样本, 看解码容差 IoU(k=0) 能否从"什么都不做"抬到 >0.3
#   抬不上去 => 实现/架构级 bug; 抬得上去 => 泛化/数据问题
set -u
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/envs/cu121/bin/python

pkill -f 'train_skelnet_[f]m' 2>/dev/null
sleep 4

run_gate () {
  N=$1
  STEPS=$2
  echo "########## 门槛测试 N=$N samples, $STEPS 步 ##########"
  $PY -u tools/train_skelnet_dit.py \
      --overfit $N --steps $STEPS --batch 64 \
      --bridge --bridge-hide-g \
      --depth 6 --hidden 256 --heads 4 --inject-layers 2 \
      --tgt-shards data/top10_style23/shards_gtskel_w7 \
      --cond-shards data/top10_style23/shards_std \
      --lr 2e-4 --eval-every 250 --es-patience 0 \
      --log logs/gate_overfit_N$N.log \
      --out assets/gate_overfit_N$N.pt 2>&1 | tail -26
}

run_gate 1 3000
run_gate 8 3000
echo "GATE_DONE"
