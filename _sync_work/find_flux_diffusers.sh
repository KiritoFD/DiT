#!/bin/bash
# 找 diffusers 格式的 FLUX VAE (省掉键名重映射)
cd /root/Workspace/xy/DiT || exit 1
for M in "black-forest-labs/FLUX.1-schnell" "AI-ModelScope/FLUX.1-schnell" "black-forest-labs/FLUX.1-dev"; do
  echo "=== $M ==="
  curl -sS -m 20 "https://www.modelscope.cn/api/v1/models/${M}/repo/files?Revision=master&Root=vae" 2>/dev/null | head -c 900
  echo
done
echo "=== 搜索名字带 flux 的仓库 ==="
curl -sS -m 20 "https://www.modelscope.cn/api/v1/dolphin/models?PageSize=15&PageNumber=1&Name=FLUX" 2>/dev/null \
  | head -c 900
echo
