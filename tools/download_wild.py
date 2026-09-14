#!/opt/conda/envs/cu121/bin/python
import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "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"

from huggingface_hub import snapshot_download

OUT = "/root/Workspace/xy/HCSU"
print(f"Downloading wild.zip parts to {OUT}...", flush=True)
t0 = time.time()

for part in range(1, 7):
    fname = f"wild.zip.{part:03d}"
    print(f"  Downloading {fname}...", flush=True)
    try:
        path = snapshot_download("Tongji209/HCSU", repo_type="dataset",
                                  local_dir=OUT, allow_patterns=[fname],
                                  token=os.environ["HF_TOKEN"])
        sz = os.path.getsize(os.path.join(OUT, fname))
        print(f"  Done: {fname} ({sz/1024/1024:.0f}MB) [{time.time()-t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"  FAILED: {fname}: {e}", flush=True)

print(f"\nAll parts downloaded in {time.time()-t0:.0f}s", flush=True)
print("Now concatenate: cat wild.zip.001 wild.zip.002 ... > wild.zip && unzip wild.zip")
