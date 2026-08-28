# -*- coding: utf-8 -*-
"""Write a fake zero-step ctrl ckpt for eval-speed tests."""
import sys, os, torch
d = sys.argv[1]
os.makedirs(d, exist_ok=True)
ck = {"train_steps": 0, "ctrl": {}}
p = os.path.join(d, "0000000.pt")
torch.save(ck, p)
open(p + ".done", "w").close()
print("wrote", p)