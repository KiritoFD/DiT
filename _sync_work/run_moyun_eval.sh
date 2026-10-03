#!/bin/bash
# GPU 评 moyun 的 5 个 ckpt
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT/ref/moyi
PY=/opt/conda/envs/cu121/bin/python
for ck in assets/results/moyun_repro/moyun_0005000.pt \
          assets/results/moyun_repro/moyun_0010000.pt \
          assets/results/moyun_repro/moyun_0015000.pt \
          assets/results/moyun_repro/moyun_0020000.pt \
          assets/results/moyun_repro/moyun_0025000.pt; do
  [ -f "$ck" ] || continue
  echo "=== $(basename $ck) ==="
  LC_ALL=C.UTF-8 $PY -u tools/moyun_eval.py --ckpt "$ck" \
    --csv assets/eval_v13_seen_fixed.csv --n 20 --steps 50 --cfg 4.0 \
    --device cuda --save-samples 2>&1 | grep -E "ssim|lpips|poster|Traceback|Error" | tail -4
done
echo "=== MOYUN EVAL DONE ==="
