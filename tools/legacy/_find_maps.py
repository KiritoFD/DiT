import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
base = "/root/Workspace/xy/DiT"
for f in os.listdir(base):
    if "index" in f.lower() or "map" in f.lower() or "final_images" in f:
        print(f)
# also check if MCCD has a manifest
mccd = os.path.join(base, "MCCD", "MCCD")
if os.path.isdir(mccd):
    for f in os.listdir(mccd):
        print("MCCD:", f)
