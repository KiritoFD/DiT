"""Fix std_image face access: specs are dicts."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')
old = "        for f in script_specs[SID[script]].faces:"
new = "        for f in script_specs[SID[script]]['faces']:"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('faces access fixed')
