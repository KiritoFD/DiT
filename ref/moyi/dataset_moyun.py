"""MultiLabelNestedDataset —— 适配我们的数据（csv + 预编码 latent shards）。

## 为什么不用 ImageFolder
ref 原版按目录结构取 (calligrapher, font, charactor) 三个标签，
数据是 `training_set/<callig>/<font>/<char>/*.png`。

我们的数据是 **csv + 预编码 latent shards**，直接返回 latent 更快
（省掉训练循环里的 VAE encode）。

## 返回 8 元组（与 train_moyun2_RF.py 的 for 循环对齐）
    image, edge, skeleton, y, stroke, callig_feature, font_feature, char_feature

    image/edge/skeleton: (4, 32, 32) float32 —— 12ch 的三段
    y:      tuple(callig_id, script_id, char_id)  各 (1,) long
    stroke: None（use_stroke=False，模型不会用）
    *_feature: (1,1) 占位（feature_dict 里用不到，走 LabelEmbedder）
"""
import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiLabelNestedDataset(Dataset):
    def __init__(self, csv_file, img_shards, edge_shards, skel_shards,
                 transform=None, char_vocab=None, num_classes=None,
                 img_root=None):
        """
        Args:
            csv_file:    训练 csv（含 old_50k_id / calligrapher_id / script_id / character_id）
            img_shards:  目标图 latent shards 目录（data/50k/shards_img）
            edge_shards: edge latent shards 目录（data/50k/shards_aux_canny）
            skel_shards: 骨架 latent shards 目录（data/50k/shards_std_fixed）
            char_vocab:  {character: idx}，None 时从 csv 现建
        """
        super().__init__()
        self.rows = list(csv.DictReader(open(csv_file, encoding="utf-8")))
        self.img_root = img_root

        # 词表
        if char_vocab is None:
            chars = sorted(set(r["character"] for r in self.rows))
            char_vocab = {c: i for i, c in enumerate(chars)}
        self.char_vocab = char_vocab

        # 三个 shard 的 id -> (文件, 偏移)
        self.img_idx = self._index(img_shards)
        self.edge_idx = self._index(edge_shards)
        self.skel_idx = self._index(skel_shards)
        self._cache = {}

        # 过滤：三个 shard 都能查到的行
        keep = []
        for r in self.rows:
            oid = r.get("old_50k_id", "").strip()
            if not oid:
                continue
            oid = int(oid)
            if oid in self.img_idx and oid in self.edge_idx and oid in self.skel_idx:
                keep.append(r)
        dropped = len(self.rows) - len(keep)
        if dropped:
            print(f"  [dataset] {len(self.rows)} -> {len(keep)} 行"
                  f"（{dropped} 行缺 shard，已丢弃）", flush=True)
        self.rows = keep

        if num_classes is None:
            num_classes = max(int(r["calligrapher_id"]) for r in self.rows) + 1
        self.num_classes = num_classes

    @staticmethod
    def _index(shards_dir):
        idx = {}
        for f in sorted(glob.glob(os.path.join(shards_dir, "*.npz"))):
            with np.load(f) as d:
                for j, iid in enumerate(d["img_ids"]):
                    idx[int(iid)] = (f, j)
        return idx

    def _lat(self, idx, oid):
        f, j = idx[oid]
        if f not in self._cache:
            self._cache[f] = np.load(f)["latents"]
        return self._cache[f][j]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        oid = int(r["old_50k_id"])
        image = torch.from_numpy(
            np.asarray(self._lat(self.img_idx, oid), dtype=np.float32))
        edge = torch.from_numpy(
            np.asarray(self._lat(self.edge_idx, oid), dtype=np.float32))
        skel = torch.from_numpy(
            np.asarray(self._lat(self.skel_idx, oid), dtype=np.float32))

        cid = int(r["calligrapher_id"]) % self.num_classes
        sid = int(r["script_id"]) % self.num_classes
        ch = r["character"]
        chid = self.char_vocab.get(ch, 0) % self.num_classes

        y = (torch.tensor([cid], dtype=torch.long),
             torch.tensor([sid], dtype=torch.long),
             torch.tensor([chid], dtype=torch.long))
        stroke = torch.zeros(1, dtype=torch.long)
        feat = torch.zeros(1, 1, dtype=torch.float32)
        return image, edge, skel, y, stroke, feat, feat, feat
