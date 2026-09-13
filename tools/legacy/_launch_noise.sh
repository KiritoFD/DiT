#!/bin/bash
cd /root/Workspace/xy/DiT
nohup /opt/conda/bin/python -u tools/vae_noise_full.py > tools/vae_noise_full.log 2>&1 &
echo "PID=$!"
