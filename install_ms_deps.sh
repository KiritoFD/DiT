#!/bin/bash
set -x
PIP=/opt/conda/bin/pip
PY=/opt/conda/bin/python

echo '=== modelscope_hub ==='
$PIP install -q modelscope_hub 2>&1 | tail -3

echo '=== full modelscope deps (still not touching torch) ==='
$PIP install -q modelscope 2>&1 | tail -8

echo '=== verify ==='
MODELSCOPE_CACHE=/root/Workspace/xy/.cache/modelscope $PY -c 'import modelscope; print("modelscope", modelscope.__version__)' 2>&1 | tail -5

echo '=== torch still intact ==='
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'