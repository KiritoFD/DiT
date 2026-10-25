"""One-off fix: the stem-expansion block in transfer.py needs a local torch import."""
from pathlib import Path

p = Path('/home/ds/Workspace/DiT/src/unet/variants/transfer.py')
s = p.read_text(encoding='utf-8')
old = "    if 'std_stem.0.weight' in source and source['std_stem.0.weight'].dim()==4 \\"
new = ("    import torch as _torch_stem\n"
       "    if 'std_stem.0.weight' in source and source['std_stem.0.weight'].dim()==4 \\")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
s = s.replace("        pad=torch.zeros(w.shape[0],2,*w.shape[2:],dtype=w.dtype)",
              "        pad=_torch_stem.zeros(w.shape[0],2,*w.shape[2:],dtype=w.dtype)")
p.write_text(s, encoding='utf-8')
print('transfer.py fixed')
