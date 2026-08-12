#!/bin/bash
cd /root/Workspace/xy/DiT
export XFORMERS_DISABLED=1
pkill -f train.py 2>/dev/null
sleep 2
mkdir -p results/overfit_500
nohup /opt/conda/bin/python train.py --config overfit.json > results/overfit_500/log.txt 2>&1 &
echo "launched pid $!"
