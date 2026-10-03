#!/bin/bash
# ??? train.py ??? VRAM: bash vr_sweep.sh <config??> <B...>
# ??? Mem: cur/peak + nvidia-smi ??, 20 ??
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
NAME=$1; shift
CFG=src/train/configs/${NAME}_pretrain.json
OUT=/tmp/vr_sweep/$NAME
mkdir -p $OUT
printf '%-10s %-6s %-18s %-10s %s\n' config B log_peak smi_peak step_s
for B in $*; do
    rm -rf "$OUT/B$B"
    ( for i in $(seq 1 300); do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$OUT/B$B.gpu" 2>/dev/null
        sleep 1
      done ) &
    SAMP=$!
    $PY -u src/train/train.py --config $CFG \
        --global-batch-size $B --max-steps 20 --log-every 1 \
        --ckpt-every 1000000 --in-mem-eval false \
        --results-dir "$OUT/B$B" > "$OUT/B$B.log" 2>&1
    RC=$?
    kill $SAMP 2>/dev/null
    MEM=$(grep -o 'Mem: [0-9.]*/' "$OUT/B$B.log" | tail -1)
    GMEM=$(sort -n "$OUT/B$B.gpu" 2>/dev/null | tail -1)
    SP=$(grep -oE 'Steps/Sec: [0-9.]+' "$OUT/B$B.log" | tail -1)
    if [ $RC -ne 0 ]; then R="FAIL($RC)"; else R=OK; fi
    printf '%-10s %-6s %-18s %-10s %s\n' "$NAME" "$B" "peak=${MEM:-?}G($R)" "${GMEM:-?}" "${SP:-?}"
done