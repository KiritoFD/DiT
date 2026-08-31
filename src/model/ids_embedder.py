# -*- coding: utf-8 -*-
"""IDS 组件码本字嵌入器

核心思路: 汉字 = 部件的递归组合。用 IDS (Ideographic Description Sequences)
把字拆成部件, 字嵌入 = 部件嵌入的扁平池化 (mean)。

优势 vs 原 LabelEmbedder(35130×384):
  - 参数量: 1571×384 ≈ 60万 (降 95.5%)
  - 零样本泛化: 没见过的字, 查 IDS 组合部件即得嵌入
  - 书体解耦: 部件是书体无关的, 书体差异全归 callig 分支
  - 草篆覆盖: 部件从真迹学, 不依赖标准字体

接口与 LabelEmbedder 完全兼容:
  forward(labels, train, force_drop_ids) -> (B, embed_dim)
"""
import os
import torch
import torch.nn as nn

# IDS 结构操作符 (12个)
IDS_OPS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")


def load_ids_map(ids_file):
    """加载 IDS 字典: char -> ids_string"""
    ids_map = {}
    with open(ids_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0].startswith("U+"):
                ch = parts[1]
                ids = parts[2]
                if ch not in ids_map:
                    ids_map[ch] = ids
    return ids_map


def extract_components(ids_str):
    """从 IDS 字符串提取叶子部件 (非操作符、非ASCII标记)"""
    return [c for c in ids_str if c not in IDS_OPS and not c.isspace() and ord(c) > 127]


def build_component_vocab(ids_map, min_freq=1):
    """构建部件词表: component -> comp_id"""
    from collections import Counter
    comp_counter = Counter()
    for ids_str in ids_map.values():
        comp_counter.update(extract_components(ids_str))
    # 按频率排序, 高频在前
    comps = sorted(comp_counter.items(), key=lambda x: -x[1])
    comp2id = {c: i for i, (c, cnt) in enumerate(comps) if cnt >= min_freq}
    return comp2id, comp_counter


class IDSCharEmbedder(nn.Module):
    """IDS 组件码本字嵌入器

    输入: char_id (int) 或 char (str)
    输出: (B, embed_dim) 字嵌入

    内部结构:
      - comp_embedding: nn.Embedding(num_components, embed_dim)
      - char_to_comps: dict char_id -> List[comp_id] (预计算, 不参与训练)
      - null_embed: nn.Parameter (CFG null token)
    """

    def __init__(self, num_characters, embed_dim, ids_file, char_id_to_char=None,
                 dropout_prob=0.0, use_cfg_embedding=True, chars_per_script=7026):
        """
        Args:
            num_characters: 字符表大小 (glyph 级, = num_scripts * chars_per_script)
            embed_dim: 输出嵌入维度
            ids_file: IDS 字典文件路径
            char_id_to_char: dict char_id -> char (char_id 是 0-based 字符索引)
            dropout_prob: CFG dropout 概率 (仅用于接口兼容, 实际 drop 在外层做)
            use_cfg_embedding: 是否使用 CFG null token
            chars_per_script: 每个书体的字符数 (用于 glyph_id -> char_id 转换)
        """
        super().__init__()
        self.num_classes = num_characters
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.use_cfg_embedding = use_cfg_embedding
        self.chars_per_script = chars_per_script

        # 加载 IDS 并构建部件词表
        ids_map = load_ids_map(ids_file)

        # 只用 char_id_to_char 中出现的字来构建部件词表 (减少部件数量)
        if char_id_to_char is not None:
            used_chars = set(char_id_to_char.values())
            ids_map_filtered = {c: ids_map[c] for c in used_chars if c in ids_map}
        else:
            ids_map_filtered = ids_map

        comp2id, comp_counter = build_component_vocab(ids_map_filtered)
        self.num_components = len(comp2id)
        self.comp2id = comp2id

        # 部件嵌入表
        self.comp_embedding = nn.Embedding(self.num_components, embed_dim)

        # 预计算 char_id -> comp_ids 映射 (buffer, 不参与训练)
        # 用 padding 到固定长度, 方便 batch 处理
        max_comps = 8  # 统计: max=7, 留 1 个余量
        self.max_comps = max_comps

        # char_id_to_char: char_id 是 0-based 字符索引 (0..chars_per_script-1)
        # 如果为 None, 假设 char_id 就是 Unicode codepoint
        if char_id_to_char is None:
            char_id_to_char = {i: chr(i) for i in range(chars_per_script)}

        # 构建 char_id -> comp_ids 的查找表 (char_id 级别, 不是 glyph_id)
        # 用两个 buffer: comp_ids (chars_per_script, max_comps) 和 comp_mask (chars_per_script, max_comps)
        comp_ids = torch.zeros(chars_per_script, max_comps, dtype=torch.long)
        comp_mask = torch.zeros(chars_per_script, max_comps, dtype=torch.bool)

        covered = 0
        for char_id in range(chars_per_script):
            ch = char_id_to_char.get(char_id)
            if ch is None:
                continue
            ids_str = ids_map.get(ch)
            if ids_str is None:
                continue
            comps = extract_components(ids_str)
            comp_ids_list = [comp2id[c] for c in comps if c in comp2id]
            if not comp_ids_list:
                continue
            covered += 1
            n = min(len(comp_ids_list), max_comps)
            comp_ids[char_id, :n] = torch.tensor(comp_ids_list[:n], dtype=torch.long)
            comp_mask[char_id, :n] = True

        self.register_buffer("char_comp_ids", comp_ids)
        self.register_buffer("char_comp_mask", comp_mask)
        self.coverage = covered / chars_per_script

        # CFG null token (可学习)
        if use_cfg_embedding:
            self.null_embed = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.null_embed = None

        # 未覆盖字的 fallback: 用所有部件的均值 (可学习)
        self.fallback_embed = nn.Parameter(torch.zeros(embed_dim))

    def token_drop(self, labels, force_drop_ids=None):
        """CFG dropout: 把部分 label 替换为 null token (num_classes)"""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        """
        Args:
            labels: (B,) glyph_id (= script_id * chars_per_script + char_id), 或 num_classes 表示 null
            train: 是否训练模式 (用于 dropout)
            force_drop_ids: (B,) 强制 drop 的 mask
        Returns:
            (B, embed_dim) 字嵌入
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        # 分离 null 和正常 label
        null_mask = (labels == self.num_classes)
        # 注意: null token (num_classes) 需要先 clamp 到合法范围, 否则 % chars_per_script 会出错
        valid_labels = labels.clamp(0, self.num_classes - 1)

        # glyph_id -> char_id: char_id = glyph_id % chars_per_script
        # 对于 null token, char_ids 会是 0, 但后面会被 null_embed 覆盖, 所以无害
        char_ids = valid_labels % self.chars_per_script

        # 查部件嵌入
        comp_ids = self.char_comp_ids[char_ids]  # (B, max_comps)
        comp_mask = self.char_comp_mask[char_ids]  # (B, max_comps)

        # 池化: mean(部件嵌入)
        comp_embs = self.comp_embedding(comp_ids)  # (B, max_comps, embed_dim)
        comp_embs = comp_embs * comp_mask.unsqueeze(-1).float()  # mask 掉 padding
        comp_sum = comp_embs.sum(dim=1)  # (B, embed_dim)
        comp_cnt = comp_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        char_emb = comp_sum / comp_cnt  # (B, embed_dim)

        # 未覆盖字用 fallback
        has_comps = comp_mask.any(dim=1, keepdim=True)  # (B, 1)
        char_emb = torch.where(has_comps, char_emb, self.fallback_embed.expand_as(char_emb))

        # null token
        if self.null_embed is not None:
            char_emb = torch.where(null_mask.unsqueeze(-1), self.null_embed.expand_as(char_emb), char_emb)

        return char_emb

    def extra_repr(self):
        return (f"num_characters={self.num_classes}, embed_dim={self.embed_dim}, "
                f"num_components={self.num_components}, coverage={self.coverage:.2%}")


# 便捷函数: 从 fame csv 构建 char_id -> char 映射
def build_char_id_map_from_csv(csv_file):
    """从训练 csv 构建 char_id -> char 映射"""
    import csv
    char_id_to_char = {}
    with open(csv_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            char_id = int(row["character_id"])
            ch = row["character"]
            char_id_to_char[char_id] = ch
    return char_id_to_char
