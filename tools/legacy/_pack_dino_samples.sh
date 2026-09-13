#!/bin/bash
# 在远程打包样本图片
cd /root/Workspace/xy/DiT
mkdir -p /tmp/dino_samples
while IFS= read -r p; do
    if [ -f "$p" ]; then
        cp "$p" /tmp/dino_samples/
    fi
done < /dev/stdin
echo "copied $(ls /tmp/dino_samples/ | wc -l) files"
tar czf /tmp/dino_samples.tar.gz -C /tmp/dino_samples .
echo "tar size: $(du -sh /tmp/dino_samples.tar.gz)"
