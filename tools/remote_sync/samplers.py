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
