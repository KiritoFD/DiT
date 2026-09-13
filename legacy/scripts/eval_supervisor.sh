#!/bin/bash
cd /root/Workspace/xy/DiT
while true; do
  /opt/conda/bin/python -u src/eval/universal_metrics_daemon.py >> /tmp/universal_daemon.log 2>&1
  echo "daemon exited, restarting in 10s" >> /tmp/universal_daemon.log
  sleep 10
done
