# -*- coding: utf-8 -*-
"""生成 canny_d3 + 全量 csv (从 shards 读全部 img_ids)."""
import subprocess, time
HOST="root@10.176.54.17"; PORT="36430"

GEN_CANNY_D3 = '''import os, sys, glob, time, csv
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.stdout.reconfigure(encoding="utf-8")
IN = "/root/Workspace/xy/DiT/final_canny"
OUT = "/root/Workspace/xy/DiT/final_canny_d3"
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(IN, "*.png")))
todo = [(f, os.path.join(OUT, os.path.basename(f))) for f in files if not os.path.exists(os.path.join(OUT, os.path.basename(f)))]
print(f"[canny_d3] {len(todo)} todo / {len(files)} total", flush=True)

import multiprocessing as mp
se = generate_binary_structure(2, 2)
def dilate(args):
    fin, fout = args
    img = np.asarray(Image.open(fin).convert("L")) > 127
    out = binary_dilation(img, structure=se, iterations=3)
    Image.fromarray((out.astype(np.uint8)*255), mode="L").save(fout)

t0 = time.time()
with mp.Pool(16) as pool:
    for done, _ in enumerate(pool.imap_unordered(dilate, todo, chunksize=256)):
        if (done+1) % 50000 == 0:
            print(f"[canny_d3] {done+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
print(f"[canny_d3] DONE {len(todo)} in {time.time()-t0:.0f}s", flush=True)

# 生成全量 csv
print("[csv] generating train_full.csv from shards...", flush=True)
SHARDS = "/root/Workspace/xy/DiT/final_latents"
all_ids = []
for sp in sorted(glob.glob(os.path.join(SHARDS, "shard_*.npz"))):
    d = np.load(sp)
    all_ids.extend(d["img_ids"].tolist())
    d.close()
all_ids.sort()
print(f"[csv] {len(all_ids)} img_ids from shards", flush=True)
out_csv = "/root/Workspace/xy/DiT/5script/train_full.csv"
with open(out_csv, "w", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image_path","calligrapher_id","script_id","glyph_id","character_id","character"])
    w.writeheader()
    for iid in all_ids:
        w.writerow({"image_path": f"final_imgs_256/{iid}.png",
                     "calligrapher_id": 0, "script_id": 0,
                     "glyph_id": 0, "character_id": 0, "character": ""})
print(f"[csv] wrote {out_csv} ({len(all_ids)} rows)", flush=True)
print("ALL_DONE", flush=True)
'''

import tempfile
with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
    f.write(GEN_CANNY_D3)
    local = f.name

r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT, local,
    "root@10.176.54.17:/tmp/_gen_canny_d3.py"],
    capture_output=True, timeout=60)
print("scp:", r.returncode==0)

def run(cmd, timeout=60):
    for i in range(5):
        try:
            r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                               capture_output=True,text=True,timeout=timeout)
            if r.returncode==0: return r.stdout
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    return None

# 停旧 GPU 训练
print("kill:", run("tmux kill-session -t structgpu 2>/dev/null; pkill -9 -f train_struct_decoder 2>/dev/null; sleep 1; echo K", timeout=30))
# 后台生成
print("launch:", run("cd /root/Workspace/xy/DiT && nohup /opt/conda/bin/python /tmp/_gen_canny_d3.py > /tmp/_gen_canny_d3.log 2>&1 & echo GEN_STARTED", timeout=30))
time.sleep(30)
print("check:", run("tail -5 /tmp/_gen_canny_d3.log", timeout=30))