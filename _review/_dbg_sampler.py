import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
from src.utils.samplers import LongEpochDistributedSampler  # noqa: E402

N, B, D = 28569, 240, 5000
full = (D * B) // N
print("full =", full, "tail =", (D * B) % N, "D*B =", D * B)


class DS:
    def __len__(self):
        return N


s = LongEpochDistributedSampler(DS(), steps_per_epoch=D, batch_size=B, seed=0)
idx = list(iter(s))
first = idx[:full * N]
print("len(idx)=", len(idx), "len(first)=", len(first))
print("zeros in first =", first.count(0), " ones =", first.count(1))
for c in range(full):
    ch = first[c * N:(c + 1) * N]
    if len(set(ch)) != N:
        print("chunk", c, "NOT a permutation; len(set)=", len(set(ch)),
              "zeros=", ch.count(0))
        break
else:
    print("all %d chunks are permutations" % full)
print("chunk0[0:5] =", first[:5])
print("chunk1[0:5] =", first[N:N + 5])
print("len(sorted(first))=", len(sorted(first)),
      "len(range*N*full)=", len(list(range(N)) * full))
sf = sorted(first)
exp = list(range(N)) * full
print("equal?", sf == exp)
for i, (x, y) in enumerate(zip(sf, exp)):
    if x != y:
        print("first mismatch at", i, "got", x, "want", y)
        break
print("idx[0:5] =", idx[:5])
g = torch.Generator()
g.manual_seed(0)
print("randperm(0)[0:5] =", torch.randperm(N, generator=g)[:5].tolist())
