"""Fix shared_baseline to early-return when metrics.json already exists."""
import ast
from pathlib import Path

for w in ('yan', 'chu'):
    p = Path(f'/home/ds/Workspace/xy/qwen-writer-res-{w}-20261025/code/writer_residual.py')
    s = p.read_text(encoding='utf-8')
    if 'if (out / \'metrics.json\').exists():' in s:
        print(w, 'already fixed')
        continue
    old = ("        out = root / 'evaluations' / 'shared_baseline'\n"
           "        out.mkdir(parents=True, exist_ok=False)\n")
    new = ("        out = root / 'evaluations' / 'shared_baseline'\n"
           "        if (out / 'metrics.json').exists():\n"
           "            print('SHARED_BASELINE cached', flush=True)\n"
           "            return\n"
           "        out.mkdir(parents=True, exist_ok=True)\n")
    assert s.count(old) == 1, (w, s.count(old))
    s = s.replace(old, new)
    p.write_text(s, encoding='utf-8')
    ast.parse(s)
    print(w, 'patched')
