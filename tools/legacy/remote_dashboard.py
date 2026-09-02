#!/usr/bin/env python
import os, glob, json, subprocess, datetime, time

# Dashboard 文件路径
DASHBOARD_DIR = '/root/Workspace/xy/DiT/tools'
POSTER = os.path.join(DASHBOARD_DIR, 'eval_poster.png')
LATEST = os.path.join(DASHBOARD_DIR, 'eval_latest.png')
REMOTE_BASE = '/root/Workspace/xy/DiT'
REMOTE_RESULTS = os.path.join(REMOTE_BASE, '5script/results')

def find_latest_ckpt_dir():
    # 找到最新的实验目录
    exp_dirs = glob.glob(os.path.join(REMOTE_RESULTS, '*'))
    latest_exp = max(exp_dirs, key=os.path.getmtime)
    run_dirs = glob.glob(os.path.join(latest_exp, '*'))
    latest_run = max(run_dirs, key=os.path.getmtime)
    ckpt_dir = os.path.join(latest_run, 'checkpoints')
    return ckpt_dir

def update_dashboard():
    ckpt_dir = find_latest_ckpt_dir()
    print(f'Latest ckpt: {ckpt_dir}')
    
    # 更新 eval_latest.png
    latest_path = os.path.join(ckpt_dir, 'eval_latest.png')
    if os.path.exists(latest_path):
        subprocess.run(['cp', latest_path, LATEST])
    
    # 更新 eval_poster.png（简化版，只显示最新的几个step）
    eval_samples_dir = os.path.join(ckpt_dir, 'eval_samples')
    if os.path.exists(eval_samples_dir):
        steps = sorted([d for d in os.listdir(eval_samples_dir) if d.startswith('step')])
        if steps:
            # 这里简化处理，直接复制最新的 eval_latest.png 作为 poster
            subprocess.run(['cp', latest_path, POSTER])
    
    print(f'Dashboard updated at {datetime.datetime.now()}')

# 每60秒更新一次
while True:
    try:
        update_dashboard()
    except Exception as e:
        print(f'Error: {e}')
    time.sleep(60)
