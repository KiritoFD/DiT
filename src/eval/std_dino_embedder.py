# -*- coding: utf-8 -*-
"""标准字形 DINO char/glyph 嵌入器（冻结查表，零可训练参数，直通）。

背景（docs/system/25_dino_embed_direct.md）
------------------------------------------
DINO 标准字形 (kai/li 干净印刷体) 特征的「外形一致性」极好（AUC 0.92-0.96），
可直接作为 char/glyph embedding 注入，**无需可训练投影扭曲信号**。

设计原则（用户明确）
------------------
- 不训练网络去扭曲初始信号 —— 特征够好就该**直通**。
- DINO 原始维度 768，模型 hidden=384。用**固定插值**（零参数）768→384 降维，
  而不是可训练 char_proj 投影。插值不引入可学习参数，也不改变已训练维度方向。
- 构建时一次性算好降维表，之后完全冻结。可训练参数 = 0（仅 CFG null token 可选）。

本模块：
- 加载 `_sync_work/std_dino_char_table_768.npy`（7026×768, char_id 级）。
- 构建时固定插值降维到 embed_dim（默认 384），存为冻结 buffer。
- 输入 glyph_id（= script_id*7026 + char_id）或 char_id，查表得 (B, embed_dim)。
- 接口与 LabelEmbedder / IDSCharEmbedder 兼容：
  `forward(labels, train, force_drop_ids) -> (B, embed_dim)`
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TABLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "_sync_work",
    "std_dino_char_table_384_pca.npy")


def _fixed_interp_table(table, out_dim):
    """固定线性插值降维 (N, 768) -> (N, out_dim)。零可学习参数，构建时一次性算。"""
    t = torch.from_numpy(table).float()          # (N, 768)
    t = t.unsqueeze(1)                            # (N, 1, 768)
    t = F.interpolate(t, size=out_dim, mode="linear", align_corners=False)  # (N,1,out_dim)
    return t.squeeze(1)                            # (N, out_dim)


class StdDinoCharEmbedder(nn.Module):
    """标准字形 DINO 冻结查表字嵌入器（固定插值降维，直通）。

    输入: glyph_id (int, = script_id*chars_per_script + char_id) 或 char_id
    输出: (B, embed_dim) 冻结的 DINO 特征向量
    """

    def __init__(self, num_characters, embed_dim, table_path=None,
                 dropout_prob=0.0, use_cfg_embedding=True, chars_per_script=7026):
        super().__init__()
        self.num_classes = num_characters
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.use_cfg_embedding = use_cfg_embedding
        self.chars_per_script = chars_per_script

        # 加载标准字形 DINO 表 (char_id 级, 768)
        path = table_path or DEFAULT_TABLE
        table = np.load(path).astype(np.float32)
        assert table.shape[0] == chars_per_script, \
            f"std dino table rows {table.shape[0]} != chars_per_script {chars_per_script}"
        d_in = table.shape[1]
        if d_in != embed_dim:
            # 固定插值降维（零参数，直通，不扭曲方向）
            table = _fixed_interp_table(table, embed_dim).numpy()
            self.downsample = f"fixed-interp {d_in}->{embed_dim}"
        else:
            self.downsample = "identity"

        # 冻结 buffer（不参与梯度）
        self.register_buffer("char_table", torch.from_numpy(table))

        # CFG null token（可选可学习，唯一可训练参数）
        if use_cfg_embedding:
            self.null_embed = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.null_embed = None

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        null_mask = (labels == self.num_classes)
        valid = labels.clamp(0, self.num_classes - 1)
        char_ids = valid % self.chars_per_script
        out = self.char_table[char_ids]  # (B, embed_dim)

        if self.null_embed is not None:
            out = torch.where(null_mask.unsqueeze(-1), self.null_embed, out)
        return out

    def extra_repr(self):
        return (f"num_characters={self.num_classes}, embed_dim={self.embed_dim}, "
                f"chars_per_script={self.chars_per_script}, "
                f"downsample={self.downsample}, trainable=0(frozen)+null")
