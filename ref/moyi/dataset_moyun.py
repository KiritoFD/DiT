"""MultiLabelNestedDataset —— 适配我们的数据（csv + 预编码 latent shards）。

## 为什么不用 ImageFolder
ref 原版按目录结构取 (calligrapher, font, charactor) 三个标签，
数据是 `training_set/<callig>/<font>/<char>/*.png`。

我们的数据是 **csv + 预编码 latent shards**，直接返回 latent 更快
（省掉训练循环里的 VAE encode）。

## 返回 8 元组（与 train_moyun2_RF.py 的 for 循环对齐）
    image, edge, skeleton, y, stroke, callig_feature, font_feature, char_feature

    image/edge/skeleton: (4, 32, 32) float32 —— 12ch 的三段
    y:      tuple(callig_id, script_id, char_id) 各 (1,) long 或 (3,) long
    stroke: None（use_stroke=False，模型不会用）
    *_feature: (1,1) 占位（feature_dict 里用不到，走 LabelEmbedder）
"""
import csv
import glob
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiLabelNestedDataset(Dataset):
    def __init__(self, csv_file, img_shards, edge_shards=None, skel_shards=None,
                 transform=None, char_vocab=None, callig_vocab=None, script_vocab=None,
                 num_classes=None, img_root=None):
        """
        Args:
            csv_file:    训练/评测 csv（含 img_id / old_50k_id / calligrapher / script / character）
            img_shards:  目标图 latent shards 目录（如 data/top10_style23/shards_img）
            edge_shards: 辅助图/edge latent shards 目录（如 data/top10_style23/shards_aux_skel3 或 shards_gtskel_w7）
            skel_shards: 骨架 latent shards 目录（如 data/top10_style23/shards_std_w7）
            char_vocab:  {character: idx}，None 时从 csv 现建
        """
        super().__init__()
        self.rows = list(csv.DictReader(open(csv_file, encoding="utf-8")))
        self.img_root = img_root

        # 1. 词表构建（确定性密集映射，杜绝哈希冲突）
        if callig_vocab is None:
            # 优先用名字，其次用 calligrapher_id
            calligs = sorted(set(r.get("calligrapher", r.get("calligrapher_id", "")) for r in self.rows if r.get("calligrapher", r.get("calligrapher_id", ""))))
            callig_vocab = {c: i for i, c in enumerate(calligs)}
        self.callig_vocab = callig_vocab

        if script_vocab is None:
            scripts = sorted(set(r.get("script", r.get("script_id", "")) for r in self.rows if r.get("script", r.get("script_id", ""))))
            script_vocab = {s: i for i, s in enumerate(scripts)}
        self.script_vocab = script_vocab

        if char_vocab is None:
            chars = sorted(set(r["character"] for r in self.rows if "character" in r))
            char_vocab = {c: i for i, c in enumerate(chars)}
        self.char_vocab = char_vocab

        # 2. 索引 shards
        self.img_idx = self._index(img_shards) if img_shards else {}
        self.edge_idx = self._index(edge_shards) if edge_shards else self.img_idx
        self.skel_idx = self._index(skel_shards) if skel_shards else self.img_idx
        self._cache = {}

        # 3. 过滤：能在 shards 中匹配到的有效行
        keep = []
        for r in self.rows:
            oid = self._extract_id(r)
            if oid is None:
                continue
            if oid in self.img_idx and oid in self.edge_idx and oid in self.skel_idx:
                r["_resolved_id"] = oid
                keep.append(r)

        dropped = len(self.rows) - len(keep)
        if dropped:
            print(f"  [dataset] {len(self.rows)} -> {len(keep)} 行（{dropped} 行缺 shard 已跳过）", flush=True)
        self.rows = keep

        # 4. 类别数确定（Embedding 尺寸）
        max_id = max(
            len(self.callig_vocab),
            len(self.script_vocab),
            len(self.char_vocab)
        )
        if num_classes is None:
            num_classes = max(6000, max_id + 10)
        self.num_classes = num_classes

    @staticmethod
    def _extract_id(r):
        for k in ("img_id", "old_50k_id", "id", "pair_id"):
            val = r.get(k, "")
            if val and str(val).strip():
                try:
                    return int(str(val).strip())
                except ValueError:
                    pass
        return None

    @staticmethod
    def _index(shards_dir):
        idx = {}
        if not shards_dir or not os.path.exists(shards_dir):
            return idx
        for f in sorted(glob.glob(os.path.join(shards_dir, "*.npz"))):
            with np.load(f) as d:
                for j, iid in enumerate(d["img_ids"]):
                    try:
                        idx[int(iid)] = (f, j)
                    except ValueError:
                        pass
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
        oid = r["_resolved_id"]

        image = torch.from_numpy(
            np.asarray(self._lat(self.img_idx, oid), dtype=np.float32))
        edge = torch.from_numpy(
            np.asarray(self._lat(self.edge_idx, oid), dtype=np.float32))
        skel = torch.from_numpy(
            np.asarray(self._lat(self.skel_idx, oid), dtype=np.float32))

        # 标签映射
        c_raw = r.get("calligrapher", r.get("calligrapher_id", ""))
        cid = self.callig_vocab.get(c_raw, int(r.get("calligrapher_id", 0)) % self.num_classes)

        s_raw = r.get("script", r.get("script_id", ""))
        sid = self.script_vocab.get(s_raw, int(r.get("script_id", 0)) % self.num_classes)

        ch = r.get("character", "")
        chid = self.char_vocab.get(ch, int(r.get("character_id", 0)) % self.num_classes)

        # y: 保持与原版和 train_moyun_ours 对齐的 tuple
        y = (torch.tensor([cid], dtype=torch.long),
             torch.tensor([sid], dtype=torch.long),
             torch.tensor([chid], dtype=torch.long))

        stroke = torch.zeros(1, dtype=torch.long)
        feat = torch.zeros(1, 1, dtype=torch.float32)
        return image, edge, skel, y, stroke, feat, feat, feat
