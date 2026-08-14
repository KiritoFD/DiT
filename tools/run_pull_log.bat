@echo off
cd /d g:\GitHub\DiT\tools
set PYTHONIOENCODING=utf-8
python pull_log.py --loop --interval 15
