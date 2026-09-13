import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("/root/Workspace/xy/DiT/mccd_mapping.json"))
print("type:", type(d).__name__)
if isinstance(d, dict):
    print("keys:", list(d.keys())[:10])
    k0 = list(d.keys())[0]
    print("sample:", k0, "->", str(d[k0])[:150])
    print("len:", len(d))
elif isinstance(d, list):
    print("len:", len(d))
    print("sample:", str(d[0])[:150])
