#!/bin/bash
echo '=== portal-x64 as root ==='
cd /root/Workspace
./portal-x64 --help 2>&1 | head -30
echo ''
echo '=== ls xy dir ==='
ls -la /root/Workspace/xy/
echo ''
echo '=== existing DiT files ==='
ls -la /root/Workspace/xy/DiT/ 2>/dev/null | head -30