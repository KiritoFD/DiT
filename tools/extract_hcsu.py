#!/opt/conda/envs/cu121/bin/python
import os, shutil
import pyzipper

BASE = "/root/Workspace/xy/HCSU"
PASSWORD = b"eWZNA9hzJoyMHtQwVh7QnBtN7"

for zf in ["bei.zip", "tie.zip"]:
    zp = os.path.join(BASE, zf)
    out_dir = os.path.join(BASE, zf.replace(".zip", ""))
    
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Extracting {zf}...", flush=True)
    try:
        with pyzipper.AESZipFile(zp, 'r') as z:
            z.setpassword(PASSWORD)
            z.extractall(out_dir)
        count = sum(1 for r, d, fs in os.walk(out_dir) for f in fs if f.lower().endswith('.png'))
        print(f"  {count} PNG files", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

# Count total
for dataset in ["bei", "tie"]:
    d = os.path.join(BASE, dataset)
    if os.path.isdir(d):
        count = sum(1 for r, dd, fs in os.walk(d) for f in fs if f.lower().endswith('.png'))
        print(f"{dataset}: {count} PNG files total")
        for r, dd, fs in os.walk(d):
            pngs = [f for f in fs if f.lower().endswith('.png')]
            if pngs:
                rel = os.path.relpath(r, d)
                print(f"  {rel}: {len(pngs)} PNGs")
                break
