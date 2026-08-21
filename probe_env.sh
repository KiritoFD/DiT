#!/bin/bash
echo '=== candidates with torch ==='
for p in python python3 /usr/bin/python3.6 /usr/bin/python3.8 /opt/conda/bin/python /usr/local/bin/python /usr/bin/python; do
  if [ -x "$p" ]; then
    v=$("$p" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)
    t=$("$p" -c 'import torch; print(torch.__version__)' 2>/dev/null)
    echo "-- $p (py $v) -- torch: ${t:-NONE}"
  fi
done
echo '=== locate torch ==='
find / -maxdepth 7 -type d -name torch 2>/dev/null | grep -v '/proc/' | head
echo '=== portal-x64 ==='
file /root/Workspace/portal-x64 2>/dev/null
echo '=== workspace content ==='
ls -la /root/Workspace/ 2>/dev/null
echo '=== any manual/readme ==='
ls /root/Workspace/*.md /root/Workspace/*.txt /root/Workspace/*help* 2>/dev/null