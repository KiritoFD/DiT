"""Handle mask=None from single-prompt encode."""
import ast
from pathlib import Path

p = Path('/home/ds/Workspace/xy/qwen-lora-ref-20261025/code/train_std.py')
s = p.read_text(encoding='utf-8')
old = """            emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
            enc_mask = torch.nn.utils.rnn.pad_sequence([e[1][0] for e in per], batch_first=True)
            img_pad = torch.nn.utils.rnn.pad_sequence([e[2][0] for e in per], batch_first=True)"""
new = """            emb = torch.nn.utils.rnn.pad_sequence([e[0][0] for e in per], batch_first=True)
            enc_mask = torch.nn.utils.rnn.pad_sequence(
                [e[1][0] if e[1] is not None else torch.ones(e[0].shape[1], dtype=torch.bool,
                                                             device=e[0].device) for e in per],
                batch_first=True)
            img_pad = torch.nn.utils.rnn.pad_sequence(
                [e[2][0] if e[2] is not None else torch.zeros(e[0].shape[1], dtype=torch.bool,
                                                              device=e[0].device) for e in per],
                batch_first=True)"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
ast.parse(s)
print('none-guard patched')
