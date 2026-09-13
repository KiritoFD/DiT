#!/bin/bash
echo '=== search DiT on modelscope ==='
timeout 20 curl -s "https://modelscope.cn/api/v1/dolphin/models?PageNumber=1&PageSize=20&Query=DiT-XL" 2>&1 | head -c 2000
echo ''
echo ''
echo '=== search directly (modelscope old API) ==='
timeout 20 curl -s "https://modelscope.cn/api/v1/models?Query=DiT-XL" 2>&1 | head -c 1500
echo ''
echo '=== try known mirror repos on hf-mirror ==='
for repo in "AI-ModelScope/DiT-XL-2-256x256" "Lokyanus/DiT-XL-2-256x256" "facebook/DiT-XL-2" "facebook/DiT-XL-2-256"; do
  code=$(timeout 12 curl -s -o /dev/null -w "%{http_code}" "https://hf-mirror.com/$repo/resolve/main/DiT-XL-2-256x256.pt")
  echo "$repo -> HTTP $code"
done