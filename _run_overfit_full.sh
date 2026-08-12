#!/bin/bash
cd /root/Workspace/xy/DiT
export XFORMERS_DISABLED=1
pkill -9 -f 'python train.py' 2>/dev/null
sleep 2
mkdir -p results/overfit_500_full
rm -rf results/overfit_500_full/*-DiT-2Cond-XL-2
nohup /opt/conda/bin/python train.py --config overfit_full.json > results/overfit_500_full/log.txt 2>&1 &
echo "launched pid $!"