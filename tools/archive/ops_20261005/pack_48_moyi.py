import os
import tarfile
import time

base_dir = "/home/ds/Workspace"
out_tar = "/tmp/moyi_history_clean.tar.gz"

dirs_to_scan = [
    os.path.join(base_dir, "moyi/results"),
    os.path.join(base_dir, "moyi/logs"),
    os.path.join(base_dir, "DiT/experiments"),
]

valid_exts = {".json", ".csv", ".log", ".txt", ".md"}

print("Scanning for experiment files (all json, csv, log, txt, and posters)...")
files_to_pack = []

for d in dirs_to_scan:
    if not os.path.exists(d):
        continue
    for root, _, filenames in os.walk(d):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            fp = os.path.join(root, f)
            rel_p = os.path.relpath(fp, base_dir)

            # Skip large binaries
            if ext in {".pt", ".pth", ".ckpt", ".npz", ".bin", ".safetensors"}:
                continue

            # Keep all json, csv, log, txt, md
            if ext in valid_exts:
                files_to_pack.append((fp, rel_p))
            # Also keep key eval posters and samples (png/jpg under 5MB)
            elif ext in {".png", ".jpg"}:
                if os.path.getsize(fp) < 5 * 1024 * 1024:
                    files_to_pack.append((fp, rel_p))

print(f"Total files collected: {len(files_to_pack)}")

t0 = time.time()
with tarfile.open(out_tar, "w:gz") as tar:
    for fp, rel_p in files_to_pack:
        tar.add(fp, arcname=rel_p)

dt = time.time() - t0
sz_mb = os.path.getsize(out_tar) / (1024**2)
print(f"Archive created: {out_tar} ({sz_mb:.2f} MB, {len(files_to_pack)} files in {dt:.1f}s)")
