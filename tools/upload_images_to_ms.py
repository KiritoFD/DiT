import os
import sys
import time
from modelscope_hub.api import HubApi

token = "ms-beb37b56-038e-455f-b4e9-81ee79e1edb4"
api = HubApi(token=token)
repo_id = "ArcherFD/CalligDiT"
repo_type = "dataset"
zip_path = "/root/Workspace/xy/MODELSCOPE_UPLOAD/images.zip"

print("=" * 75)
print(
    f"Starting upload of images.zip ({os.path.getsize(zip_path):,} bytes) to {repo_id}..."
)
print("=" * 75)

t0 = time.time()
res = api.upload_file(
    repo_id=repo_id,
    repo_type=repo_type,
    path_or_fileobj=zip_path,
    path_in_repo="images.zip",
    commit_message="Add images.zip containing all 393,486 cleaned 256x256 images",
)
dt = time.time() - t0
print("=" * 75)
print(f"Upload completed in {dt:.1f}s ({dt/60:.2f} mins)!")
print("Commit Result:", res)
print("=" * 75)
