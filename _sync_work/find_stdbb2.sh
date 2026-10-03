#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== 全部 results 目录 (v12 及以后) ==="
ls -1 assets/results/ | grep -E '^v[0-9]' | sort -V
echo
echo "=== MANIFEST 里提到 std/stdskel 的行 ==="
grep -iE "std|骨架|skel" assets/results/MANIFEST.md 2>/dev/null | head -20
echo
echo "=== 各 run 的 skel 条件 + model (v20 以后) ==="
for f in $(ls -1d assets/results/v2*/ 2>/dev/null); do
  cfg=$(ls "$f"resolved_config.json 2>/dev/null)
  [ -z "$cfg" ] && continue
  /opt/conda/envs/cu121/bin/python -c "
import json
try:
    c=json.load(open('$cfg'))
    s=c.get('skel_latent_shards_dir','-')
    if 'std' in str(s) or 'aux' in str(s) or 'gtskel' in str(s):
        print(f\"  {'$f'.strip('/')}: skel={s}  model={c.get('model','-')}  glyph_inject={c.get('glyph_inject_layers','-')}  lr={c.get('lr','-')}\")
except Exception:
    pass
" 2>/dev/null
done
