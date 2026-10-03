#!/bin/bash
# D6 剩余 2 个配置 —— 并行加速（显存只用了 667MiB / 24GB，完全够）
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/eval_v13_strict.csv
LOGD=_sync_work/d6_logs
mkdir -p $LOGD
B=assets/results

# v15a（可训表 87-pair）
if [ ! -f assets/d6_v15a_multistyle.json ]; then
  $PY tools/probe_cond_causality.py \
    --ckpt $B/v15a_multistyle_k4/20260919-223715-v15a-multistyle-k4-pool/checkpoints/0150000.pt \
    --eval-csv $CSV --cfg 1.0 --n-chars 16 --steps 50 \
    --out assets/d6_v15a_multistyle.json > $LOGD/d6_v15a_multistyle.log 2>&1 &
  P1=$!
else P1=""; fi

# v15c（表最可分但 dmod 最低）
if [ ! -f assets/d6_v15c_fixed.json ]; then
  $PY tools/probe_cond_causality.py \
    --ckpt $B/v15c_fixed/20260921-212920-v15c-multistyle-k4-ctx/checkpoints/0210000.pt \
    --eval-csv $CSV --cfg 1.0 --n-chars 16 --steps 50 \
    --out assets/d6_v15c_fixed.json > $LOGD/d6_v15c_fixed.log 2>&1 &
  P2=$!
else P2=""; fi

[ -n "$P1" ] && wait $P1 && echo "[D6] v15a done"
[ -n "$P2" ] && wait $P2 && echo "[D6] v15c done"

echo "[D6-P2] ALL DONE"
ls -la assets/d6_*.json
