#!/bin/bash
kill $(ps aux | grep vae_noise_gpu | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 1
echo "killed"
