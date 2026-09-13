import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("/root/Workspace/xy/DiT/pretrained_models/dino_embeddings/glyph_dino_index.json"))
print("type:", type(d).__name__)
if isinstance(d, dict):
    print("keys:", list(d.keys())[:5])
    k0 = list(d.keys())[0]
    print("sample:", k0, "->", str(d[k0])[:80])
    print("len:", len(d))
