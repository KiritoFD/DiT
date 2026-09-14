#!/opt/conda/envs/cu121/bin/python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import hf_hub_download
p = hf_hub_download("Tongji209/HCSU", "README.md", repo_type="dataset")
txt = open(p).read()
start = txt.find("Archive Passwords")
end = txt.find("Repository File Tree")
print(txt[start:end])
