#!/opt/conda/envs/cu121/bin/python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import HfApi
api = HfApi()
files = list(api.list_repo_tree("Tongji209/HCSU", repo_type="dataset", recursive=True))
for f in files:
    print(f.path)
