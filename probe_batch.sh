#!/bin/bash
# Quick VRAM probe for s16/s17 @240 (20s sampling)
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/opt/conda/bin/python

probe_one () {
    local CFG=$1
    local TAG=$2
    local BS=$3
    local TMP_CFG=/tmp/${TAG}_probe.json
    $PY -c "
import json
d = json.load(open('$CFG'))
d['global_batch_size'] = $BS
d['max_steps'] = 5
d['ckpt_every'] = 0
d['auto_eval'] = False
json.dump(d, open('$TMP_CFG','w'))
"
    echo "=== PROBE $TAG bs=$BS ==="
    rm -f /tmp/${TAG}_smi.log
    nohup $PY train.py --config $TMP_CFG > /tmp/${TAG}_probe.log 2>&1 &
    local TPID=$!
    sleep 4
    for i in $(seq 1 20); do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> /tmp/${TAG}_smi.log
        sleep 1
    done
    kill $TPID 2>/dev/null
    wait $TPID 2>/dev/null
    sleep 2
    pkill -9 -f train.py 2>/dev/null
    sleep 2
    local PEAK=$(sort -n /tmp/${TAG}_smi.log | tail -1)
    echo "[$TAG bs=$BS] peak=${PEAK}MiB (target <=23000)"
}

probe_one "s16_s_ddpm_test.json"  "s16" 240
probe_one "s17_s_flow_test.json"  "s17" 240
echo "=== DONE ==="
for t in s16 s17; do
    PEAK=$(sort -n /tmp/${t}_smi.log | tail -1)
    echo "[$t] peak VRAM = ${PEAK} MiB"
done