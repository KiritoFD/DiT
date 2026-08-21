#!/bin/bash
set -x
PY=/opt/conda/bin/python
PIP=/opt/conda/bin/pip

echo '=== conda python & pip ==='
$PY --version
$PIP --version

echo '=== torch in conda ==='
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

echo '=== pytorch version pin note ==='
echo 'Do NOT upgrade torch. Only install modelscope and its deps.'

echo '=== pip index check ==='
$PIP config list 2>/dev/null | grep -i index || echo 'no pip index config'

echo '=== install modelscope (no deps on torch) ==='
$PIP install -q --no-deps modelscope 2>&1 | tail -5
echo '--- modelscope deps (safe subset) ---'
$PIP install -q requests tqdm filelock sortedcontainers simplejson addict attrs 2>&1 | tail -3

echo '=== verify import ==='
MODELSCOPE_CACHE=/root/Workspace/xy/.cache/modelscope $PY -c 'import modelscope; print("modelscope", modelscope.__version__)'