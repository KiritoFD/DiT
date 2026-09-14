#!/opt/conda/envs/cu121/bin/python
import os, sys, time, json, csv
from collections import Counter, defaultdict
from PIL import Image
import numpy as np

BASE = "/root/Workspace/xy/HCSU"

# Step 1: Concatenate wild.zip parts
wild_parts = sorted([f for f in os.listdir(BASE) if f.startswith("wild.zip.")])
wild_combined = os.path.join(BASE, "wild.zip")
if not os.path.exists(wild_combined):
    print("Concatenating wild.zip parts...", flush=True)
    with open(wild_combined, "wb") as out:
        for part in wild_parts:
            path = os.path.join(BASE, part)
            print(f"  Appending {part} ({os.path.getsize(path)/1024/1024:.0f}MB)...", flush=True)
            with open(path, "rb") as inp:
                while True:
                    chunk = inp.read(1024*1024*16)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"wild.zip: {os.path.getsize(wild_combined)/1024/1024/1024:.1f}GB", flush=True)

# Step 2: Unzip
for zf in ["bei.zip", "tie.zip", "wild.zip"]:
    out_dir = os.path.join(BASE, zf.replace(".zip", ""))
    if not os.path.isdir(out_dir):
        print(f"Unzipping {zf}...", flush=True)
        os.system(f"cd {BASE} && unzip -q -o {zf} -d {out_dir}")
        print(f"  Done: {out_dir}", flush=True)
    else:
        print(f"{zf} already unzipped: {out_dir}", flush=True)

# Step 3: Analyze
for dataset in ["bei", "tie", "wild"]:
    d = os.path.join(BASE, dataset)
    if not os.path.isdir(d):
        print(f"\n{dataset}: NOT FOUND")
        continue
    
    # Walk directory tree
    png_count = 0
    subdirs = []
    for root, dirs, files in os.walk(d):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if pngs:
            rel = os.path.relpath(root, d)
            subdirs.append((rel, len(pngs)))
            png_count += len(pngs)
    
    print(f"\n{'='*50}")
    print(f"{dataset}: {png_count} PNG images in {len(subdirs)} directories")
    print(f"{'='*50}")
    
    # Sample a few directories
    for rel, cnt in sorted(subdirs)[:5]:
        print(f"  {rel}: {cnt} images")
    if len(subdirs) > 5:
        print(f"  ... +{len(subdirs)-5} more dirs")
    
    # Sample an image
    for rel, cnt in subdirs[:1]:
        sample_dir = os.path.join(BASE, dataset, rel)
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".png")][:3]
        for sf in sample_files:
            fp = os.path.join(sample_dir, sf)
            img = Image.open(fp)
            print(f"  Sample: {sf} size={img.size} mode={img.mode}")

# Step 4: Detailed per-calligrapher stats
print(f"\n{'='*50}")
print("Detailed per-calligrapher stats:")
print(f"{'='*50}")
for dataset in ["bei", "tie", "wild"]:
    d = os.path.join(BASE, dataset)
    if not os.path.isdir(d):
        continue
    
    cal_counts = defaultdict(int)
    for root, dirs, files in os.walk(d):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if pngs:
            # Try to extract calligrapher name from path
            rel = os.path.relpath(root, d)
            parts = rel.split("/")
            if len(parts) >= 1:
                cal_name = parts[0]
                cal_counts[cal_name] += len(pngs)
    
    print(f"\n{dataset}: {len(cal_counts)} calligraphers/dirs, {sum(cal_counts.values())} total images")
    for cal, cnt in sorted(cal_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cal}: {cnt}")
    if len(cal_counts) > 10:
        print(f"  ... +{len(cal_counts)-10} more")
