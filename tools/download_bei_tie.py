#!/opt/conda/envs/cu121/bin/python
import os, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import snapshot_download

OUT = "/root/Workspace/xy/HCSU"
t0 = time.time()
for fname in ["bei.zip", "tie.zip"]:
    print(f"Downloading {fname}...", flush=True)
    snapshot_download("Tongji209/HCSU", repo_type="dataset",
                      local_dir=OUT, allow_patterns=[fname],
                      token=os.environ["HF_TOKEN"])
    sz = os.path.getsize(os.path.join(OUT, fname))
    print(f"  Done: {fname} ({sz/1024/1024:.0f}MB) [{time.time()-t0:.0f}s]", flush=True)
print(f"All done in {time.time()-t0:.0f}s", flush=True)
