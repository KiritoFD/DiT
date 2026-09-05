# -*- coding: utf-8 -*-
"""
dit_skel_1cond.py — DiT-1CondSkel: 无 char 条件的骨架引导生成模型（独立文件，不影响 DiT_2Cond）。

设计 (用户定方向: 不做两阶段、从头开始、不用 char 条件、直接拿 skel latent 当输入):
  * **输入 concat**: x = cat([img_latent, skel_latent], dim=1) → (B, 8, 32, 32)
    一次 patch embed (8→D)，skel 作为**输入的一部分**（不是 ControlNet 注入）
  * **条件只有书家**: adaLN 仅由 y_callig 调制（风格）
  * **无 char 表**: 无 y_char_embedder/char_proj/char_scale —— 字形身份由 skel latent 承载
  * CFG: 训练时 skel 按 prob 置零（skel_drop_prob）+ callig dropout（标准 4-way 简化版）
  * REPA 接口保留: return_intermediate_layers 捕获 block 中间特征（与公共 src/loss/repa 兼容）

forward(x, skel, t, y_callig, return_intermediate_layers=None, ...)：
    x: (N,4,32,32) noised latent; skel: (N,4,32,32) GT skeleton latent
    返回 (out, {layer: feats}) 或 out
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules as M
from .dit import TimestepEmbedder, LabelEmbedder
try:
    from .patchembed import PatchEmbed
except Exception:
    PatchEmbed = None  # fallback 自实现


class _PatchEmbedFallback(nn.Module):
    """输入 (N,C,H,W) → patch embed → (N,T,D)。独立实现避免依赖。"""
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = None

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.proj(x)          # (N, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)


class DiT_1CondSkel(nn.Module):
    """单条件(书家) + skel-latent 输入拼接的扩散 Transformer。"""

    def __init__(self, *, num_calligraphers, input_channels=8, hidden_size=384,
                 depth=12, num_heads=6, patch_size=2, learn_sigma=False,
                 cond_drop_prob=0.10, skel_drop_prob=0.10,
                 norm_type="rms", mlp_type="swiglu", qk_norm=True,
                 rope=True, rope_theta=100.0, attn_impl="sdpa",
                 use_checkpoint=False, **unused):
        super().__init__()
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.out_channels = 4 if not learn_sigma else 8
        self.in_channels = 4
        self.patch_size = patch_size
        self.cond_drop_prob = cond_drop_prob
        self.skel_drop_prob = skel_drop_prob

        # 输入: 8 通道 = img_latent(4) + skel_latent(4)
        self.x_embedder = PatchEmbed(input_channels, hidden_size, patch_size) \
            if PatchEmbed is not None else _PatchEmbedFallback(input_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # 书家条件 (唯一 adaLN 调制)
        self.y_callig_embedder = LabelEmbedder(num_calligraphers, hidden_size,
                                               dropout_prob=0.0,
                                               use_cfg_embedding=True)
        self.callig_proj = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        # 可学习幅度(风格调制强度)
        self.callig_scale = nn.Parameter(torch.tensor(1.0))

        # POS_EMB (learnable, 兼容 patch 网格)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, hidden_size) * 0.02)

        self.blocks = nn.ModuleList([
            M.DiTBlock(hidden_size, num_heads, mlp_ratio=4.0, qk_norm=qk_norm,
                       norm_type=norm_type, mlp_type=mlp_type,
                       attn_impl=attn_impl)
            for _ in range(depth)])
        self.final_layer = M.FinalLayer(hidden_size, patch_size, self.out_channels,
                                        norm_type=norm_type)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # 书家表
        nn.init.normal_(self.y_callig_embedder.embedding_table.weight, std=0.02)
        if hasattr(self.y_callig_embedder, 'null_embed') and self.y_callig_embedder.null_embed is not None:
            nn.init.normal_(self.y_callig_embedder.null_embed, std=0.02)
        if hasattr(self.x_embedder, 'initialize_weights'):
            self.x_embedder.initialize_weights()
        # FinalLayer zero-init (恒等起点)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y_callig, skel=None, return_intermediate_layer=None,
                return_intermediate_layers=None):
        """x:(N,4,32,32) noised; skel:(N,4,32,32) GT skel latent (None=无条件);
        t:(N,); y_callig:(N,)。签名与 DiT_2Cond 一致 (模型调用方用 **model_kwargs 透传)。"""
        # 训练时 skel dropout (CFG: 无条件时无骨架)
        if skel is None:
            skel = torch.zeros(x.shape[0], 4, x.shape[2], x.shape[3], device=x.device)
        if self.training and self.skel_drop_prob > 0:
            drop = torch.rand(skel.shape[0], device=skel.device) < self.skel_drop_prob
            if drop.any():
                skel = torch.where(drop.view(-1, 1, 1, 1).expand_as(skel),
                                   torch.zeros_like(skel), skel)
        # callig dropout
        if self.training and self.cond_drop_prob > 0:
            drop_c = torch.rand(y_callig.shape[0], device=y_callig.device) < self.cond_drop_prob
            y_callig = torch.where(drop_c, self.y_callig_embedder.num_classes, y_callig)

        # 输入拼接: (N, 8, H, W) → tokens
        x_in = torch.cat([x, skel], dim=1)
        x = self.x_embedder(x_in)
        n_tokens = x.shape[1]
        pe = self.pos_embed[:, :n_tokens]
        x = x + pe.to(x.device)

        # 书家 adaLN 条件
        e_callig = self.y_callig_embedder(y_callig, False)
        y_emb = self.callig_scale * self.callig_proj(e_callig)
        t_emb = self.t_embedder(t)
        c = t_emb + y_emb

        # 多层捕获 (REPA 兼容)
        _repa_layers = None
        if return_intermediate_layers is not None:
            _repa_layers = (return_intermediate_layers if isinstance(
                return_intermediate_layers, (list, tuple)) else (return_intermediate_layers,))
            intermediate = {}
        else:
            intermediate = None
        _single = return_intermediate_layer

        if self.training and False:  # 保留 checkpoint 语法兼容
            pass
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if _repa_layers is not None and i in _repa_layers:
                intermediate[i] = x
            elif _single is not None and i == _single:
                intermediate = x
        x = self.final_layer(x, c)
        x = self.unpatchify(x)   # tokens → (N, out_channels, 32, 32)
        if _repa_layers is not None:
            return x, intermediate
        if _single is not None:
            return x, intermediate
        return x


def skel_1cond_model(name="DiT-1CondSkel-S/2", num_calligraphers=1013, **kw):
    """模型工厂 (独立, 不注册进 DiT_2Cond_models 避免污染)。"""
    if "S/2" in name:
        return DiT_1CondSkel(num_calligraphers=num_calligraphers,
                             hidden_size=384, depth=12, num_heads=6, patch_size=2, **kw)
    if "S/1" in name:
        return DiT_1CondSkel(num_calligraphers=num_calligraphers,
                             hidden_size=384, depth=12, num_heads=6, patch_size=1, **kw)
    raise NotImplementedError(name)