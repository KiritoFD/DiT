import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = "/root/Workspace/xy/DiT/tools/vae_noise_results"
# check progress files
for f in os.listdir(d):
    fp = os.path.join(d, f)
    if "progress" in f:
        p = json.load(open(fp))
        print(f, p)
# check detail
det = os.path.join(d, "detail_f4_kl-f4_p4.json")
if os.path.exists(det):
    j = json.load(open(det))
    print("detail entries:", len(j))
    if j:
        print("  last:", j[-1])
