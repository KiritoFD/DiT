"""Sampling utilities for sparse multi-factor condition training."""

import math
from collections import Counter

import torch
from torch.utils.data import Sampler


class DistributedFactorBalancedSampler(Sampler):
    """Tempered inverse-frequency sampling over character and calligrapher.

    Full inverse-frequency sampling badly repeats singleton rows. Exponents below
    one flatten the long tail while preserving useful frequency information.
    A single deterministic global draw is strided across ranks, matching the
    partitioning behavior of DistributedSampler.
    """

    def __init__(self, dataset, num_replicas=1, rank=0, seed=0,
                 char_alpha=0.5, callig_alpha=0.25):
        if not hasattr(dataset, "samples"):
            raise TypeError("factor-balanced sampling requires dataset.samples")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        def _glyph(row):
            return int(row.get("glyph_id", row["character_id"]))

        char_count = Counter(_glyph(row) for row in dataset.samples)
        callig_count = Counter(int(row["calligrapher_id"]) for row in dataset.samples)
        weights = []
        for row in dataset.samples:
            char_freq = char_count[_glyph(row)]
            callig_freq = callig_count[int(row["calligrapher_id"])]
            weights.append((char_freq ** -float(char_alpha))
                           * (callig_freq ** -float(callig_alpha)))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.weights /= self.weights.mean()

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator)
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def summary(self):
        return {
            "weight_min": float(self.weights.min()),
            "weight_mean": float(self.weights.mean()),
            "weight_max": float(self.weights.max()),
        }


class LongEpochDistributedSampler(Sampler):
    """epoch = **固定步数**(而非"数据集一遍") —— epoch 退化为 ckpt/eval 的对齐刻度。

    动机 (2026-09-16): 原先 epoch = len(dataset)//batch = 119 步(~21s), 每 119 步
    DataLoader 就要重建一次迭代器; 而 ckpt + in-mem eval 落在 train_steps % ckpt_every
    上, 两者互不对齐 -> iterator reset 成为一份**独立**代价, 并让 Steps/Sec 出现锯齿。

    索引流: 本类吐出的就是「len(dataset) 大小的标准 randperm 的连续拼接」, 只把切分点
    从「每 len(dataset) 个样本」改成「每 steps_per_epoch * batch_size 个样本」:
      * 每个 randperm 仍是全量无放回置换, 种子逐个递增, 不重复不遗漏
        -> 采样分布与 DistributedSampler「永不 reset 地连着跑」完全一致;
      * 迭代器 reset 频率 1/119 步 -> 1/steps_per_epoch 步;
      * 令 steps_per_epoch == ckpt_every, reset 点与 ckpt+eval 点重合, 代价被吸收。
    ⚠ 语义变化(有意): "epoch" 不再等于"数据集一遍"。依赖该假设的逻辑在本项目里不存在
    (LR 调度走 max_steps, 早停走 ckpt_every)。详见 train.py --epoch-steps 的说明。
    """

    def __init__(self, dataset, steps_per_epoch, batch_size,
                 num_replicas=1, rank=0, seed=0, start_inner_epoch=0):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.set_steps(steps_per_epoch)
        # 续训: 把内部 randperm 计数器推到"已消耗的份数", 让数据流接着走而非重放。
        self._inner_epoch = int(start_inner_epoch)

    def set_steps(self, steps):
        """设定本 epoch 的步数(每 rank)。num_samples / total_size 随之更新。"""
        self.steps = int(steps)
        self.num_samples = self.steps * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def set_start_inner_epoch(self, n):
        self._inner_epoch = int(n)

    def inner_epoch_for_step(self, global_step):
        """把"已训练步数"换算成"已消耗的 randperm 份数", 供续训对齐数据流。"""
        n = len(self.dataset)
        if n <= 0:
            return 0
        return (int(global_step) * self.batch_size * self.num_replicas) // n

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """仅为兼容 DistributedSampler 的调用点而存在: 数据流按样本连续推进, 不按
        epoch 重置 —— 这正是本类的目的。"""
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        need = self.total_size
        pos = 0            # 已产出样本在"全局连续流"中的位置 (DDP 按位置分片)
        g = torch.Generator()
        while need > 0:
            g.manual_seed(self.seed + self._inner_epoch)
            self._inner_epoch += 1
            perm = torch.randperm(n, generator=g)
            take = int(min(need, perm.numel()))
            chunk = perm[:take].tolist()
            need -= take
            if self.num_replicas == 1:
                yield from chunk
            else:
                for j in range(take):
                    if (pos + j) % self.num_replicas == self.rank:
                        yield chunk[j]
            pos += take
