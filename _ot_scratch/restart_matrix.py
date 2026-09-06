#!/usr/bin/env python3
"""restart_matrix.py — 杀 skel_follow 进程, 重跑矩阵。"""
import subprocess, time

subprocess.run(['pkill', '-f', 'skel_follow_gpu'], capture_output=True)
subprocess.run(['pkill', '-f', 'run_follow_matrix'], capture_output=True)
time.sleep(3)
print('killed')

# 重跑矩阵
subprocess.Popen(['bash', '_sync_work/run_follow_matrix.sh'],
                 stdout=open('/tmp/follow_matrix.log', 'w'), stderr=subprocess.STDOUT)
print('relaunched')