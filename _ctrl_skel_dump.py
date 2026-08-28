import json, glob, os
os.chdir("/root/Workspace/xy/DiT")
# ctrl_skel eval files - show ALL fields
fs = sorted(glob.glob("5script/results/ctrl_skel/*/checkpoints/eval_auto_*.json"))
print(f"ctrl_skel eval files: {len(fs)}")
for f in fs[:3]:
    d = json.load(open(f))
    print(f"\n--- {os.path.basename(f)} ---")
    print(json.dumps(d, indent=2))
