#!/usr/bin/env bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%T') shards ==="
ls data/50k/shards_instskel20/ | head
python3 - <<'PY'
import glob, numpy as np
fs = sorted(glob.glob("data/50k/shards_instskel20/shard_*.npz"))
print("n_shards", len(fs))
if fs:
    d = np.load(fs[0])
    print("latents", d["latents"].shape, d["latents"].dtype)
PY

if pgrep -f 'src.train.train' >/dev/null; then
  echo "有训练在跑, 不启动"
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

CK=assets/eval_isolated/glyph_gate_hard_15k/0015000.pt
echo "=== $(date '+%T') train head ==="
exec $PY -u tools/train_style_head.py --ckpt "$CK" --steps 3000 --batch 128 \
    --lr 1e-4 --device cuda --out assets/style_head_instskel20.pt
