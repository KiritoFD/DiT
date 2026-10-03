#!/bin/bash
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
echo "=== v25_stdskel 目录 ==="
ls -1 assets/results/v25_stdskel/ | head
for f in $(ls -1 assets/results/v25_stdskel/*/resolved_config.json 2>/dev/null | head -2); do
  echo "--- $f"
  $PY _sync_work/dump_cfg.py "$f" | grep -iE \
    'model|skel|glyph|cond|lr|batch|ema|warmup|weight_decay|norm|mlp|rope|qk|diffusion|flow|t_sampler|shift|steps|noise|patch|deform|schedule|optimizer|attn|no_char|callig|script|spatial|sampler'
done
