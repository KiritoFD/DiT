import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = "/root/Workspace/xy/DiT/tools/vae_noise_results/detail_f4_kl-f4_p4.json"
if os.path.exists(p):
    d = json.load(open(p))
    print(len(d), "entries")
    print("last:", d[-1])
else:
    print("not found")
# summary
sp = "/root/Workspace/xy/DiT/tools/vae_noise_results/vae_noise_summary.json"
if os.path.exists(sp):
    s = json.load(open(sp))
    print("summary entries:", len(s))
    for e in s:
        print(" ", e.get("name"), "mse=", e.get("mse_mean"), "ssim=", e.get("ssim_mean"))
