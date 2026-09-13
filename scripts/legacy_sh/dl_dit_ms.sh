#!/bin/bash
set -x
export MODELSCOPE_CACHE=/root/Workspace/xy/.cache/modelscope
cd /root/Workspace/xy/DiT
# use the modelscope CLI entry point
/opt/conda/bin/modelscope download --model AI-ModelScope/DiT-XL-2-256x256 --local_dir pretrained_models 2>&1 | tail -25
echo '=== result ==='
find pretrained_models -maxdepth 2 -iname '*DiT*' -exec ls -la {} \; 2>/dev/null