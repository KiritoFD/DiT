#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%T') stop previous ==="
for p in $(pgrep -f 'src.train.train'); do
  kill "$p" 2>/dev/null || true
done
sleep 8
if pgrep -f 'src.train.train' >/dev/null; then
  echo "停不掉"
  exit 1
fi
echo "=== $(date '+%T') train 2000 ==="
$PY -u -m src.train.train --config src/train/configs/v17_lca_x.json \
    > /tmp/lca_x2.log 2>&1
echo "train exit=$?"

CK=$(ls -t assets/results/v17_lca_x/*/checkpoints/*.pt | head -1)
echo "ckpt $CK"
grep -E 'non-finite|cosine schedule' /tmp/lca_x2.log | head -3
echo "=== probe ==="
$PY tools/probe_lca.py --ckpt "$CK" --device cpu
