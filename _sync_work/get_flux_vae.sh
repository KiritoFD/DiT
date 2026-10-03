#!/bin/bash
# 取 FLUX 的 ae.safetensors (ModelScope: muse/flux_vae) —— 远端能直连就直连
cd /root/Workspace/xy/DiT || exit 1
OUT=data/pretrained/flux_vae
mkdir -p "$OUT"

echo "=== 远端 modelscope 可达性 ==="
CODE=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' https://www.modelscope.cn 2>/dev/null)
echo "modelscope -> ${CODE:-TIMEOUT}"
CODE2=$(curl -sS -m 12 -o /dev/null -w '%{http_code}' https://hf-mirror.com 2>/dev/null)
echo "hf-mirror  -> ${CODE2:-TIMEOUT}"

if [ "$CODE" != "200" ] && [ "$CODE" != "302" ]; then
  echo "[结论] 远端不可达 modelscope -> 需本机下载后 scp"
  exit 2
fi

echo
echo "=== 下载 ==="
for F in ae.safetensors configuration.json README.md; do
  URL="https://www.modelscope.cn/api/v1/models/muse/flux_vae/repo?Revision=master&FilePath=${F}"
  curl -L -sS -m 1200 -o "$OUT/$F" "$URL" || { echo "$F 下载失败"; exit 3; }
  printf '  %-20s %s bytes\n' "$F" "$(stat -c%s "$OUT/$F" 2>/dev/null)"
done

echo
echo "=== 校验 (期望 ae.safetensors = 335304388, sha256 afc8e28272cd...) ==="
stat -c%s "$OUT/ae.safetensors"
sha256sum "$OUT/ae.safetensors"
echo "--- configuration.json ---"; cat "$OUT/configuration.json"; echo
echo "--- README.md ---"; cat "$OUT/README.md"; echo
