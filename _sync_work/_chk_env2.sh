#!/bin/bash
echo "=== 各环境 python 路径 ==="
for p in /opt/conda/bin/python /opt/conda/envs/cu121/bin/python /opt/conda/envs/torch2/bin/python; do
  if [ -x "$p" ]; then
    echo "--- $p ---"
    "$p" - <<'PY'
import sys, torch
print("  python", sys.version.split()[0])
print("  torch", torch.__version__, "cuda", torch.version.cuda)
print("  cudnn", torch.backends.cudnn.version())
print("  sdpa:", hasattr(torch.nn.functional, "scaled_dot_product_attention"))
try:
    import torch._dynamo
    print("  compile/dynamo: OK")
except Exception as e:
    print("  compile/dynamo:", str(e)[:80])
try:
    import xformers
    print("  xformers", xformers.__version__)
except Exception as e:
    print("  xformers:", str(e)[:80])
PY
  else
    echo "--- $p --- (missing)"
  fi
done
echo ""
echo "=== nvcc / driver ==="
nvcc --version 2>/dev/null | tail -2
nvidia-smi | grep -a 'CUDA Version' | head -1
