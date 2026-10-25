"""Fix writer_residual.py: fp32 before shared-load, freeze/cast after verify. Idempotent."""
import ast
from pathlib import Path

for w in ('yan', 'chu'):
    p = Path(f'/home/ds/Workspace/xy/qwen-writer-res-{w}-20261025/code/writer_residual.py')
    s = p.read_text(encoding='utf-8')
    if 'pp.data = pp.data.float()' in s:
        print(w, 'already fixed')
        continue
    old = ("                      adapter_name='shared')\n"
           "    def strip_adapter(d):")
    new = ("                      adapter_name='shared')\n"
           "    for n, pp in model.named_parameters():\n"
           "        if pp.requires_grad:\n"
           "            pp.data = pp.data.float()\n"
           "    def strip_adapter(d):")
    assert s.count(old) == 1, (w, 'anchor1', s.count(old))
    s = s.replace(old, new)
    old2 = ("    for p in model.parameters():\n"
            "        p.requires_grad_(False)\n"
            "        if p.dtype == torch.float32:\n"
            "            p.data = p.data.to(torch.bfloat16)")
    new2 = ("    for n, p in model.named_parameters():\n"
            "        p.requires_grad_(False)\n"
            "        if 'lora_A.shared' in n or 'lora_B.shared' in n:\n"
            "            p.data = p.data.to(torch.bfloat16)")
    assert s.count(old2) == 1, (w, 'anchor2', s.count(old2))
    s = s.replace(old2, new2)
    p.write_text(s, encoding='utf-8')
    ast.parse(s)
    print(w, 'patched')
