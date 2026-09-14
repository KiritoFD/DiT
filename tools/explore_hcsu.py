#!/opt/conda/envs/cu121/bin/python
import os, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi
api = HfApi()
files = list(api.list_repo_tree("Tongji209/HCSU", repo_type="dataset", recursive=True))

# Separate dirs vs files
dirs = [f.path for f in files if f.path.endswith("/")]
files_only = [f for f in files if not f.path.endswith("/")]

print(f"Total items: {len(files)}")
print(f"Directories: {len(dirs)}")
print(f"Files: {len(files_only)}")

# Count unique calligraphers & scripts
calligraphers = set()
scripts = set()
for d in dirs:
    parts = d.split("/")
    if len(parts) >= 3:
        name_script = parts[-1]
        if "-" in name_script:
            cal, script = name_script.rsplit("-", 1)
            calligraphers.add(cal)
            scripts.add(script)

print(f"\nUnique calligraphers: {len(calligraphers)}")
print(f"Scripts: {sorted(scripts)}")
print(f"Calligraphers ({len(calligraphers)}): {sorted(calligraphers)}")

# Files breakdown
exts = {}
for f in files_only:
    ext = os.path.splitext(f.path)[1]
    exts[ext] = exts.get(ext, 0) + 1
print(f"\nFile types: {exts}")

# Total size
total = sum(f.size for f in files_only if hasattr(f, 'size') and f.size)
print(f"Total size: {total/1024/1024/1024:.2f} GB")
