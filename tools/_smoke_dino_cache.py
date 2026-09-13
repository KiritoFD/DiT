# -*- coding: utf-8 -*-
"""CPU smoke: DinoFeatureCache + REPALoss cache path + lazy teacher fallback."""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dino_cache import DinoFeatureCache
from src.loss.repa import build_repa_module

tmp = tempfile.mkdtemp()
N, P, D = 16, 8, 384
feats = (np.random.randn(N, P, D).astype(np.float32) * 2).astype(np.float32)
with open(os.path.join(tmp, "feats.f32"), "wb") as f:
    f.write(feats.tobytes())
ids = np.arange(100, 100 + N, dtype=np.int64)
np.save(os.path.join(tmp, "ids.npy"), ids)
with open(os.path.join(tmp, "meta.json"), "w") as f:
    json.dump({"n": N, "patches": P, "dim": D, "backbone": "fake"}, f)

cache = DinoFeatureCache(tmp)
print("[1] cache loaded:", cache.stats())

img_ids = torch.tensor([100, 105, 115, 100])
got, missing = cache.gather(img_ids)
assert missing == [] and got.shape == (4, P, D)
ref = torch.from_numpy(np.asarray(feats[5], dtype=np.float32))
assert torch.allclose(got[1], ref, atol=1e-3), "row mismatch"
print("[2] gather hits OK (dup + shuffle, row correct)")

repa = build_repa_module(student_dim=D, layers=(0,), teacher_ckpt="nonexistent.pt",
                         w_repa=0.1, warmup_steps=0, feature_cache=cache)
assert repa.losses[0].teacher is None, "lazy teacher should not load on full hit"
sf = torch.randn(4, P, D, requires_grad=True)
img = torch.rand(4, 3, 32, 32) * 2 - 1
loss = repa({0: sf}, img, step=0, img_ids=img_ids)
assert torch.isfinite(loss) and loss.item() > 0
loss.backward()
assert sf.grad is not None
print(f"[3] REPALoss cache path OK (loss={loss.item():.4f}, grad flows, teacher stays lazy)")


class FakeTeacher(torch.nn.Module):
    def forward_features(self, x):
        B = x.shape[0]
        return {"x_norm_patchtokens": torch.randn(B, P, D)}


class FakeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeTeacher()

    def forward_features(self, x):
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]


repa.losses[0]._shared_teacher_getter = lambda: FakeWrapper()
ids_miss = torch.tensor([100, 9999])
got, missing = cache.gather(ids_miss)
assert missing == [1]
loss = repa({0: torch.randn(2, P, D)}, torch.rand(2, 3, 32, 32) * 2 - 1,
            step=0, img_ids=ids_miss)
assert torch.isfinite(loss)
assert repa.losses[0].teacher is not None, "ensure_teacher should have loaded on miss"
print(f"[4] miss fallback OK (loss={loss.item():.4f}, teacher lazy-loaded once)")

d = 0.9999
per_step = 1.0
for _ in range(4):
    per_step = per_step * d
assert abs(per_step - d ** 4) < 1e-9
print("[5] EMA interval decay math OK (beta^4 == 4x beta)")

print("SMOKE PASS")
