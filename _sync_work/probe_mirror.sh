#!/bin/bash
# 远端可达性: hf-mirror / modelscope / github / pypi 各测 8 秒
for U in https://hf-mirror.com https://www.modelscope.cn https://github.com https://pypi.org https://huggingface.co; do
  R=$(curl -sS -m 8 -o /dev/null -w "%{http_code}" "$U" 2>/dev/null || echo TIMEOUT)
  printf '  %-32s -> %s\n' "$U" "$R"
done
echo
echo "=== hf-mirror 上找 flux vae ==="
curl -sS -m 12 "https://hf-mirror.com/api/models?search=flux2_vae&limit=10" 2>/dev/null | head -c 600
echo
curl -sS -m 12 "https://hf-mirror.com/api/models/ai-toolkit/flux2_vae" 2>/dev/null | head -c 1500
echo
echo "=== modelscope 上找 flux vae ==="
curl -sS -m 12 "https://www.modelscope.cn/api/v1/dolphin/models?PageSize=10&PageNumber=1&Name=flux2_vae" 2>/dev/null | head -c 800
echo
echo "=== 顺带: 远端已缓存了什么 VAE ==="
ls ~/.cache/huggingface/hub/ 2>/dev/null | head -20
find / -maxdepth 6 -iname '*flux*' -not -path '*/proc/*' 2>/dev/null | head -10
