"""Drop the resident text encoder from the eval pipe (only prompt_embeds path is used)."""
import ast
from pathlib import Path

for w in ('yan', 'chu'):
    p = Path(f'/home/ds/Workspace/xy/qwen-writer-res-{w}-20261025/code/writer_residual.py')
    s = p.read_text(encoding='utf-8')
    if 'text_encoder=None, dtype=torch.bfloat16' in s:
        print(w, 'already fixed')
        continue
    old = ("    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,\n"
           "                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')")
    new = ("    pipe = QwenImage21Pipeline.from_pretrained(root / 'base', transformer=model,\n"
           "                                               text_encoder=None,\n"
           "                                               dtype=torch.bfloat16, local_files_only=True).to('cuda')")
    assert s.count(old) == 1, (w, s.count(old))
    s = s.replace(old, new)
    p.write_text(s, encoding='utf-8')
    ast.parse(s)
    print(w, 'patched')
