#!/bin/bash
echo '=== network test ==='
echo '--- modelscope ---'
timeout 10 curl -sI https://www.modelscope.cn 2>&1 | head -1 || echo 'TIMEOUT/FAIL'
echo '--- fbaipublicfiles (DiT weights) ---'
timeout 10 curl -sI https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt 2>&1 | head -1 || echo 'TIMEOUT/FAIL'
echo '--- pypi ---'
timeout 10 curl -sI https://pypi.org/simple/ 2>&1 | head -1 || echo 'TIMEOUT/FAIL'
echo '--- github ---'
timeout 10 curl -sI https://github.com 2>&1 | head -1 || echo 'TIMEOUT/FAIL'
echo ''
echo '=== pretrained_models existing ==='
ls -la /root/Workspace/xy/DiT/pretrained_models/ 2>/dev/null
echo ''
echo '=== dataset dir ==='
ls -la /root/Workspace/xy/DiT/dataset/ 2>/dev/null | head
echo ''
echo '=== env proxy vars ==='
env | grep -i proxy || echo 'no proxy vars'