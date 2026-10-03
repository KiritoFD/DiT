#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== dino_embeddings dir ==="
ls -la data/pretrained/dino_embeddings/ 2>/dev/null
echo "=== npy shapes ==="
/opt/conda/bin/python - <<'PY'
import numpy as np, json, os
base="data/pretrained/dino_embeddings"
for f in sorted(os.listdir(base)):
    p=os.path.join(base,f)
    if f.endswith('.npy'):
        try:
            d=np.load(p); print(f, d.shape, d.dtype)
        except Exception as e:
            print(f, "ERR", e)
    elif f.endswith('.json'):
        try:
            j=json.load(open(p,encoding='utf-8')); print(f, "keys", list(j.keys())[:10])
            if 'glyphs' in j: print("  n_glyphs", len(j['glyphs']))
        except Exception as e:
            print(f, "ERR", e)
PY
echo "=== done ==="
