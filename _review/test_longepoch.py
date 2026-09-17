"""LongEpochDistributedSampler 行为验证 (纯 CPU, 本地可跑)。

验证 4 件事:
  1. len(sampler) == epoch_steps * batch_size  -> DataLoader 一个 epoch 正好 epoch_steps 步
  2. 索引流 == 「标准 randperm 的连续拼接」   -> 无放回、不重复、不遗漏、覆盖全量
  3. 多次 iter() 不重放                      -> 换 epoch(即换 DataLoader 迭代器)后数据继续推进
  4. 续训对齐                                -> inner_epoch_for_step(step) 让数据流接得上

跑法: python _review/test_longepoch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from src.utils.samplers import LongEpochDistributedSampler  # noqa: E402

N = 28569
B = 240
D = 5000          # epoch 步数 (= ckpt_every)
fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


class DS:
    def __len__(self):
        return N


print("=" * 74)
print(f"LongEpochDistributedSampler  N={N} B={B} epoch_steps={D}")
print("=" * 74)

ds = DS()
s = LongEpochDistributedSampler(ds, steps_per_epoch=D, batch_size=B,
                                num_replicas=1, rank=0, seed=0)

print("\n[1] 长度 / 步数")
check("len(sampler) == epoch_steps*batch", len(s) == D * B, f"{len(s)} == {D*B}")
from torch.utils.data import DataLoader  # noqa: E402


class LDS(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return torch.tensor(i)


ld = DataLoader(LDS(), batch_size=B, sampler=s, num_workers=0, drop_last=True)
check("len(loader) == epoch_steps", len(ld) == D, f"{len(ld)} == {D}")

print("\n[2] 索引流 = 标准 randperm 的连续拼接")
idx = list(iter(s))
check("共吐出 epoch_steps*batch 个索引", len(idx) == D * B, str(len(idx)))
full = (D * B) // N
tail = (D * B) % N
first = idx[:full * N]
from collections import Counter  # noqa: E402
# ⚠ 不能用 sorted(first) == list(range(N))*full: 左边是 [0]*full + [1]*full + ...,
#   右边是 [0,1,...,N-1]*full — 同一个多重集、不同排列。要比较多重集本身。
check("前 %d 个整份覆盖全量(不重不漏)" % full,
      Counter(first) == Counter({i: full for i in range(N)}))
check("每一份各自都是完整置换", all(
    len(set(first[c * N:(c + 1) * N])) == N for c in range(full)))
check("尾部 %d 个是第 %d 份 randperm 的前缀" % (tail, full),
      len(set(idx[full * N:])) == tail)
check("尾部无重复", len(set(idx[full * N:])) == tail)


def ref_stream(seed, units):
    """参考实现: units 份标准 randperm 首尾相接。"""
    g = torch.Generator()
    out = []
    for j in range(units):
        g.manual_seed(seed + j)
        out.extend(torch.randperm(N, generator=g).tolist())
    return out


check("与参考实现逐元素一致 (seed=0, 43 份)",
      idx[:full * N] == ref_stream(0, full)[:full * N])

print("\n[3] 多次 iter() 不重放 (换 epoch 后数据继续推进)")
a = list(iter(s))[:5]
b = list(iter(s))[:5]
check("第二次 iter() 的头部 != 第一次头部", a != b, f"{a} vs {b}")
units = full + (1 if tail else 0)   # 43: 每次 iter() 消耗的 randperm 份数
# ⚠ 计数: 第 61 行的 idx = list(iter(s)) 已经消耗掉第 1 份块, 故
#   line 87 的 a 是第 2 块头部, line 88 的 b 是第 3 块头部 -> 累计 3*units。
check("_inner_epoch 已推进", s._inner_epoch == 3 * units,
      f"_inner_epoch={s._inner_epoch} (want {3 * units})")
s2 = LongEpochDistributedSampler(ds, steps_per_epoch=D, batch_size=B, seed=0)
# b 是第 3 块头部 = 连续流中第 2*units 份 randperm 的头 5 个。
check("第三次 iter() 接在第 %d 份之后 (不重放)" % (2 * units),
      b == ref_stream(0, 2 * units + 1)[2 * units * N:2 * units * N + 5],
      f"{b} vs {ref_stream(0, 2 * units + 1)[2 * units * N:2 * units * N + 5]}")

print("\n[4] 续训对齐 inner_epoch_for_step")
for step in (0, 5000, 55000, 400000):
    got = s2.inner_epoch_for_step(step)
    want = (step * B) // N
    check(f"inner_epoch_for_step({step}) == {want}", got == want, f"got {got}")
s3n = s.inner_epoch_for_step(55000)
s3 = LongEpochDistributedSampler(ds, steps_per_epoch=D, batch_size=B, seed=0,
                                 start_inner_epoch=s3n)
check("续训流 == 连续流在 step 55000 处的切片",
      list(iter(s3))[:20] == ref_stream(0, s3n + 43)[s3n * N:s3n * N + 20],
      f"inner={s3n}")

print("\n" + "=" * 74)
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
print("=" * 74)
sys.exit(1 if fails else 0)
