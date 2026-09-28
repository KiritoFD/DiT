# -*- coding: utf-8 -*-
"""Latent-shard dataset for FontDiffuser-latent on top10_style23.

All tensors are precomputed project latents (4x32x32, x0.18215):
  target : data/top10_style23/shards_img   via img_id (same shards as v24)
  style  : same shards, a random OTHER row of the same slot (training) or the
           deterministic refs.json pool (sampling)
  content: baseline/data/content_font/deng_shards via img_id = codepoint(char)

No pixel tensors exist anywhere in the model path.
"""
import csv
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latent_common import (CONTENT_SHARDS, DIT_ROOT, TARGET_SHARDS,  # noqa: E402
                           ShardIndex)

TRAIN_CSV = f"{DIT_ROOT}/assets/train_top10_style23.csv"


class FontLatentDataset(torch.utils.data.Dataset):
    def __init__(self, rows=None, target_shards=TARGET_SHARDS,
                 content_shards=CONTENT_SHARDS, device="cpu"):
        if rows is None:
            with open(TRAIN_CSV, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.rows = rows
        self.tgt = ShardIndex(target_shards, device)
        self.con = ShardIndex(content_shards, device)
        # slot -> [(img_id, char), ...]
        self.slot_rows = {}
        for r in self.rows:
            self.slot_rows.setdefault(r["slot_name"], []).append(
                (int(r["img_id"]), r["character"]))
        self.rng = random.Random(123)

    def __len__(self):
        return len(self.rows)

    def sample_refs(self, slot, exclude_img_id, k=1):
        pool = [(i, c) for i, c in self.slot_rows[slot] if i != exclude_img_id]
        return self.rng.sample(pool, min(k, len(pool)))

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_id = int(r["img_id"])
        char = r["character"]
        slot = r["slot_name"]
        target = self.tgt.get(img_id)
        content = self.con.get(ord(char))
        ref_id, _ = self.sample_refs(slot, img_id, k=1)[0]
        style = self.tgt.get(ref_id)
        return {"target_latent": target, "content_latent": content,
                "style_latent": style, "char": char, "slot": slot}


class CollateFN:
    def __call__(self, batch):
        return {
            "target_latent": torch.stack([b["target_latent"] for b in batch]),
            "content_latent": torch.stack([b["content_latent"] for b in batch]),
            "style_latent": torch.stack([b["style_latent"] for b in batch]),
            "char": [b["char"] for b in batch],
            "slot": [b["slot"] for b in batch],
        }
