#!/bin/bash
cd /root/Workspace/xy/DiT
export XFORMERS_DISABLED=1
pkill -9 -f 'python train.py' 2>/dev/null
sleep 2
rm -rf results/full_3cond/*-DiT-3Cond-S-2
mkdir -p results/full_3cond
nohup /opt/conda/bin/python train.py --config train_full_3cond.json > results/full_3cond/log.txt 2>&1 < /dev/null &
disown
echo "launched pid $!"