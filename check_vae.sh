#!/bin/bash
echo '=== VAE files ==='
find /root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema -type f -exec ls -la {} \; 2>/dev/null
echo ''
echo '=== VAE dir depth ==='
find /root/Workspace/xy/DiT/pretrained_models/sd-vae-ft-ema -maxdepth 3 2>/dev/null
echo ''
echo '=== modelscope cache ==='
ls -la /root/Workspace/xy/.cache/modelscope/ 2>/dev/null
echo ''
echo '=== try downloading DiT-XL-2-256x256.pt from modelscope ==='
echo '(checking if exists on modelscope via API)'
timeout 15 curl -s "https://modelscope.cn/api/v1/models?Name=DiT-XL-2" 2>&1 | head -c 500