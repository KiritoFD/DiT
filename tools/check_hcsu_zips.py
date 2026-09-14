#!/opt/conda/envs/cu121/bin/python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import HfApi
api = HfApi()

# Check zip files
files = list(api.list_repo_tree("Tongji209/HCSU", repo_type="dataset", recursive=True))
zips = [f for f in files if f.path.endswith(".zip") or ".zip." in f.path]
print("ZIP/part files:")
for f in zips:
    print(f"  {f.path}: {f.size/1024/1024:.1f}MB")

# Check assets
assets = [f for f in files if f.path.startswith("assets/")]
print("\nAssets:")
for f in assets:
    print(f"  {f.path}: {f.size/1024/1024:.1f}MB" if hasattr(f, 'size') else f"  {f.path}")

# Check total unique chars per calligrapher
import json
from huggingface_hub import hf_hub_download
bei = json.load(open(hf_hub_download("Tongji209/HCSU", "bei_annotations.json", repo_type="dataset")))
tie = json.load(open(hf_hub_download("Tongji209/HCSU", "tie_annotations.json", repo_type="dataset")))

from collections import defaultdict
cal_chars = defaultdict(set)
for item in bei + tie:
    cal_chars[item["calligrapher"]].add(item["char"])

print("\nChars per calligrapher:")
for cal in sorted(cal_chars.keys()):
    print(f"  {cal}: {len(cal_chars[cal])} unique chars")
