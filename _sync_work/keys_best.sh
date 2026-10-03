#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
F=src/train/configs/v10b_stdskel_fame3_c41x_cos_e.json
echo "=== _comment 前 12 行 ==="
$PY - "$F" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print((c.get("_comment") or "")[:1600])
print("\n=== 键 ===")
for k in sorted(c):
    if k == "_comment":
        continue
    print(f"  {k} = {c[k]}")
EOF
