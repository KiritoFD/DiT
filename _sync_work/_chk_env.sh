#!/bin/bash
echo "=== conda envs ==="
/opt/conda/bin/conda env list 2>/dev/null || conda env list 2>/dev/null
echo ""
echo "=== cu121 env ==="
/opt/conda/envs/cu121/bin/python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("has sdpa:", hasattr(torch.nn.functional, "scaled_dot_product_attention"))
try:
    import torch._dynamo
    print("torch._dynamo (compile): OK")
except Exception as e:
    print("torch._dynamo:", e)
try:
    import xformers
    print("xformers", xformers.__version__)
    import xformers.ops as xo
    print("xformers fmha ops:", [n for n in dir(xo) if "attention" in n.lower()])
except Exception as e:
    print("xformers:", e)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn:", e)
PY
echo ""
echo "=== current env ==="
/opt/conda/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import xformers
    print("xformers", xformers.__version__)
except Exception as e:
    print("xformers:", e)
PY
