#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust deployment script to sync code changes to remote 4090 server.
"""

import os
import sys
import tarfile
import base64
import subprocess
import time

REMOTE_HOST = "root@10.176.54.17"
REMOTE_PORT = "36430"
REMOTE_DIR = "/root/Workspace/xy/DiT"
LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def main():
    tar_path = os.path.join(LOCAL_DIR, "patch_sync.tar.gz")
    print(f"[deploy] Packing files from {LOCAL_DIR}...")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(os.path.join(LOCAL_DIR, "models.py"), arcname="models.py")
        tar.add(os.path.join(LOCAL_DIR, "sample.py"), arcname="sample.py")
        tar.add(os.path.join(LOCAL_DIR, "src"), arcname="src")
        tar.add(os.path.join(LOCAL_DIR, "tests"), arcname="tests")
        for sub in ["analysis", "evaluation", "dashboard", "controlnet"]:
            sub_dir = os.path.join(LOCAL_DIR, "tools", sub)
            if os.path.exists(sub_dir):
                tar.add(sub_dir, arcname=f"tools/{sub}")

    tar_size = os.path.getsize(tar_path)
    print(f"[deploy] Archive created: {tar_size} bytes")

    with open(tar_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("ascii")

    remote_python = f"""
import base64, sys, os
b64_str = sys.stdin.read().strip()
raw = base64.b64decode(b64_str)
target_tar = "{REMOTE_DIR}/patch_sync.tar.gz"
with open(target_tar, "wb") as f:
    f.write(raw)
os.system("tar -xzf " + target_tar + " -C {REMOTE_DIR}/ && rm -f " + target_tar)
print("Remote extraction complete!")
"""

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
        "-p", REMOTE_PORT, REMOTE_HOST,
        f"/opt/conda/bin/python -c '{remote_python}'"
    ]

    success = False
    for attempt in range(1, 4):
        print(f"[deploy] Attempt {attempt}: sending payload to {REMOTE_HOST}:{REMOTE_PORT}...")
        try:
            p = subprocess.Popen(
                ssh_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            out, err = p.communicate(input=b64_data, timeout=60)
            if p.returncode == 0:
                print(f"[deploy] Success! Output: {out.strip()}")
                success = True
                break
            else:
                print(f"[deploy] Failed with returncode {p.returncode}: {err.strip()}")
                time.sleep(3)
        except Exception as e:
            print(f"[deploy] Exception: {e}")
            time.sleep(3)

    if os.path.exists(tar_path):
        os.remove(tar_path)

    if success:
        print("[deploy] Verifying remote files...")
        check_cmd = [
            "ssh", "-p", REMOTE_PORT, REMOTE_HOST,
            f"ls -ld {REMOTE_DIR}/src {REMOTE_DIR}/tools/analysis {REMOTE_DIR}/sample.py"
        ]
        cp = subprocess.run(check_cmd, capture_output=True, text=True)
        print(cp.stdout)

if __name__ == "__main__":
    main()
