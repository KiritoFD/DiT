# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.checkpoint import checkpoint

# 现代化组件（RMSNorm / SwiGLU / 2D-RoPE / QK-Norm / SDPA / PatchEmbed / DiTBlock / FinalLayer）
# DiT_2Cond 使用。
#
# 注：本文件曾同时存在原版 DiT（单条件）与 DiT_3Cond（三条件），二者依赖
# timm 的 PatchEmbed/Attention/Mlp。2026-08-31 清理时一并删除 —— 它们已废弃，
# 且删除后本文件不再依赖 timm（少一个重量级依赖）。
from . import modules as M


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob, use_cfg_embedding=None):
        super().__init__()
        if use_cfg_embedding is None:
            use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        # 冻结字符表时，CFG null token 需要"单独"保持可学习 —— 见 freeze_table()。
        # 为 None 表示未冻结（整表可训练，最后一行本来就参与训练）。
        self.null_embed = None

    def freeze_table(self):
        """冻结 [0, num_classes) 的字符表，但让 CFG null token 保持可学习。

        ⚠ 不能写成::

            w.requires_grad_(False)
            w[-1].requires_grad_(True)     # no-op!

        ``w[-1]`` 是索引产生的**非叶子**张量，对它的 ``requires_grad_(True)``
        是静默 no-op（已实测：调用后 ``w[-1].requires_grad`` 仍为 False）。
        旧代码正是这么写的，因此 null token 实际一直被冻结在 N(0,0.02)。

        null token 在 4-way dropout 里被大量使用（cond_drop_one 25% +
        cond_drop_all 5%），是 CFG uncond 分支的核心，理应可训练。
        这里把它拆成独立的 ``nn.Parameter``，forward 里用 ``torch.where`` 覆盖，
        零额外拷贝开销。
        """
        w = self.embedding_table.weight
        w.requires_grad_(False)
        self.null_embed = nn.Parameter(w[self.num_classes].detach().clone())
        return self.null_embed

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        out = self.embedding_table(labels)
        if self.null_embed is not None:
            # 用可学习的 null 参数覆盖最后一行（整表已冻结，否则整表都会更新）
            null_mask = (labels == self.num_classes)
            if null_mask.any():
                out = torch.where(null_mask.unsqueeze(-1), self.null_embed, out)
        return out


#################################################################################
#                                 Core DiT Model                                #
#################################################################################






def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


















class DiT_2Cond(nn.Module):
    """
    Diffusion model with a Transformer backbone conditioned on 2 discrete labels:
      1. Calligrapher (y_callig)  -> style / calligrapher identity
      2. Character   (y_char)     -> the text content (treated as a discrete class)
    Body dims are sized like DiT-XL so that DiT-XL-2-256x256.pt main weights load.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_calligraphers=1000,
        num_characters=1000,
        learn_sigma=True,
        use_checkpoint=True,
        condition_fusion="legacy",
        callig_embed_dim=None,
        char_embed_dim=None,
        cond_drop_all_prob=0.05,
        cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=0.5,
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=0.4,
        # 标准字形条件的逐层注入层数。0 = 关闭（只用输入层 token-add，旧行为）；
        # >0 = 在该数量的 block 后注入（均匀分布），与 ControlNet 的
        # ZeroAdaLNInjection 完全对齐。见下方 glyph_embedder 处的说明。
        # 显存：每层约 +150MB（batch=192 时），12 层约 +1.8G，注意 OOM。
        glyph_inject_layers=0,
        # 标准字形条件的训练期随机丢弃概率。0 = 不丢弃。
        # 作用见 forward() 中的注释：防门控 + 模拟草/篆无标准字形的真实缺失。
        glyph_drop_prob=0.0,
        char_proj_mode="full",
        freeze_char_table=False,
        # ---- IDS 组件码本字嵌入 (替代 LabelEmbedder) ----
        use_ids_char_embedder=False,  # 是否用 IDS 组件码本
        ids_file=None,                # IDS 字典文件路径
        char_id_to_char=None,         # dict char_id -> char (None 时假设 char_id=Unicode)
        # ---- 标准字形 DINO 字嵌入 (冻结查表, 零可训练参数) ----
        use_std_dino_char_embedder=False,  # 是否用标准字形 DINO 冻结表
        std_dino_table_path=None,          # 标准字形 DINO 表路径 (默认 _sync_work/std_dino_char_table_768.npy)
        chars_per_script=7026,             # 每个书体字符数 (glyph_id = script*chars_per_script + char_id)
        # ---- 现代化开关（v2 arch）----
        # 默认全部开启。全部关闭时与旧实现数值等价（同 seed 可复现旧结果）。
        norm_type="rms",        # "rms" | "layer"
        mlp_type="swiglu",      # "swiglu" | "gelu"
        qk_norm=True,
        rope=True,              # 2D axial RoPE；False 时退回固定 2D sin-cos 加到 x
        rope_theta=100.0,
        attn_impl="sdpa",       # "sdpa" | "eager"
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.qk_norm = bool(qk_norm)
        self.rope = bool(rope)
        self.rope_theta = float(rope_theta)
        self.attn_impl = attn_impl
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition_fusion = condition_fusion
        self.cond_drop_all_prob = float(cond_drop_all_prob)
        self.cond_drop_one_prob = float(cond_drop_one_prob)
        self.cond_drop_which_glyph_prob = float(cond_drop_which_glyph_prob)
        self.skel_head_enabled = bool(skel_head_enabled)
        self.use_glyph_cond = bool(use_glyph_cond)
        self.glyph_scale_init = float(glyph_scale_init)
        self.char_proj_mode = char_proj_mode
        self.freeze_char_table = bool(freeze_char_table)
        if self.cond_drop_all_prob < 0 or self.cond_drop_one_prob < 0:
            raise ValueError("condition dropout probabilities must be non-negative")
        if self.cond_drop_all_prob + self.cond_drop_one_prob > 1:
            raise ValueError("cond_drop_all_prob + cond_drop_one_prob must be <= 1")

        # 用 modules 版（自带 RMSNorm/SwiGLU/RoPE/QK-Norm），不再依赖 timm。
        self.x_embedder = M.PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if condition_fusion == "factorized_add":
            # 二因子可组合条件（V3-A）：calligrapher（风格）× glyph（内容=script×char 合并类）。
            # 每个因子独立低维 embedding + 独立投影后相加，未见的 (callig, glyph) 组合
            # 由两个各自训练充分的边际 score 组合而成，而不是靠一整张联合表 memorization。
            callig_embed_dim = callig_embed_dim or hidden_size
            char_embed_dim = char_embed_dim or hidden_size
            self.y_callig_embedder = LabelEmbedder(
                num_calligraphers, callig_embed_dim, 0.0, use_cfg_embedding=True)
            if use_std_dino_char_embedder:
                # 标准字形 DINO 冻结查表: 零可训练参数, 外形一致性 AUC>0.92
                # (docs/system/25_dino_embed_direct.md)。char_embed_dim 应=DINO 维度(768)。
                from .std_dino_embedder import StdDinoCharEmbedder
                self.y_char_embedder = StdDinoCharEmbedder(
                    num_characters, char_embed_dim, std_dino_table_path,
                    dropout_prob=0.0, use_cfg_embedding=True,
                    chars_per_script=chars_per_script)
            elif use_ids_char_embedder:
                # IDS 组件码本: 字嵌入 = 部件嵌入池化, 参数量降 95.5%, 零样本泛化
                from .ids_embedder import IDSCharEmbedder
                self.y_char_embedder = IDSCharEmbedder(
                    num_characters, char_embed_dim, ids_file, char_id_to_char,
                    dropout_prob=0.0, use_cfg_embedding=True)
            else:
                self.y_char_embedder = LabelEmbedder(
                    num_characters, char_embed_dim, 0.0, use_cfg_embedding=True)
            self.callig_proj = nn.Sequential(nn.LayerNorm(callig_embed_dim),
                                             nn.Linear(callig_embed_dim, hidden_size))
            if char_proj_mode == "ln_only":
                # DINO 384 直通：char_embed_dim == hidden_size 时，char_proj 只做
                # LayerNorm 归一化，不再 Linear 投影（省 384*384≈147K 冗余参数）。
                #
                # ⚠ 实测问题（见 docs/system/12_dino_diagnosis_20260829.md）：
                # 冻结 DINO 表的**有效秩只有 34.1 / 384**（PC1 占 26.3% 能量），
                # 83% 的最近邻落在同一书体，跨书体字符检索 top-1 仅 1.9%。
                # 也就是说字符分支拿到的信号里"书体"远多于"字符身份"。
                # 此时 char_proj 只有 LayerNorm 的 2×384 个参数，
                # **没有任何可学习容量去放大/重组那 34 个有用维度**。
                # 新配置请优先用 "mlp" 或 "full"。
                assert char_embed_dim == hidden_size, \
                    f"char_proj_mode='ln_only' requires char_embed_dim==hidden_size (got {char_embed_dim} vs {hidden_size})"
                self.char_proj = nn.LayerNorm(char_embed_dim)
            elif char_proj_mode == "mlp":
                # 推荐模式：给字符分支真正的可学习容量。
                # LayerNorm -> Linear -> SiLU -> Linear，参数量 ~2*D*D（S/2 上 +295K）。
                # 输入是近乎低秩的冻结 DINO 向量，一个非线性投影能把有用的那几个
                # 方向摊到整个 hidden 维上，而不是让 adaLN 直接吃一个 3 维流形。
                self.char_proj = nn.Sequential(
                    nn.LayerNorm(char_embed_dim),
                    nn.Linear(char_embed_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            else:
                self.char_proj = nn.Sequential(nn.LayerNorm(char_embed_dim),
                                               nn.Linear(char_embed_dim, hidden_size))
            # ── 书家/字条件的可学习幅度平衡 ───────────────────────────────
            #
            # 实测（tools/probe_condition_injection.py，s21 best ckpt，64 字）：
            #     DINO 表输出 e_char 的区分度（不同字余弦相似度） = 0.0817  <- 很好
            #     char_proj 输出                                  = 0.1189  <- 很好
            #     与书家相加后的 y_emb                            = 0.6252  <- 被淹没
            # 即 **DINO 信号本身不坏，是被书家分支的幅度压过去了**：
            #     ||callig_proj 输出|| = 12.90     ||char_proj 输出|| = 7.32
            #     比值 1.76，相加后书家主导 -> 字符区分度从 0.12 劣化到 0.63。
            #
            # 边际贡献也一致（对 adaLN shift 的相对变化）：
            #     callig 0.226   char 0.068   （字条件只有书家的 1/3）
            #
            # 这里给两个分支各一个可学习标量，让模型自行收敛到合适比例，
            # 而不是把比例硬编码成 1:1（书家与字的最优权重未必相等）。
            # 初值 1.0 保持与原实现等价，不会破坏已有 ckpt 的语义。
            self.callig_scale = nn.Parameter(torch.tensor(1.0))
            self.char_scale = nn.Parameter(torch.tensor(1.0))
            self.cond_fusion = None
        elif condition_fusion == "xl_highdim":
            # XL 高维条件：callig(384) + glyph(768) concat -> MLP -> hidden(1152)，c = t_emb + y_emb。
            # 关键认知修正：ImageNet 预训练的 adaLN/final_layer 是"分类→调制"耦合，与书法正交，
            # 因此 train.py 里会把它们重置从头学。这里只保留高维条件结构，条件向量由训练
            # 目标自行建立语义。y_scale 可学习缩放初值 ~1.0，让 y_emb 初始幅度接近 t_emb，
            # 保证 adalaN(已重置) 早期稳定，同时允许模型自由扩大/缩小条件表达。
            d_c = max(callig_embed_dim or (hidden_size // 3), 64)
            d_g = max(char_embed_dim or (hidden_size - d_c), 64)
            self.y_callig_embedder = LabelEmbedder(
                num_calligraphers, d_c, 0.0, use_cfg_embedding=True)
            self.y_char_embedder = LabelEmbedder(
                num_characters, d_g, 0.0, use_cfg_embedding=True)
            self.callig_proj = None
            self.char_proj = None
            self.cond_fusion = nn.Sequential(
                nn.LayerNorm(d_c + d_g),
                nn.Linear(d_c + d_g, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.y_scale = nn.Parameter(torch.tensor(0.05))  # y_emb 初始 norm~1.0，可学习放大
            self._y_scale_enabled = True
        else:
            self.y_callig_embedder = LabelEmbedder(num_calligraphers, hidden_size, class_dropout_prob)
            self.y_char_embedder = LabelEmbedder(num_characters, hidden_size, class_dropout_prob)
            self.cond_fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.callig_proj = self.char_proj = None

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            M.DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                       norm_type=norm_type, mlp_type=mlp_type, qk_norm=qk_norm,
                       attn_impl=attn_impl)
            for _ in range(depth)
        ])
        self.final_layer = M.FinalLayer(hidden_size, patch_size, self.out_channels,
                                        norm_type=norm_type)

        # ---- 2D axial RoPE 缓存 ----
        # persistent=False：不写进 state_dict，避免任何 ckpt key 变化。
        self.num_patches = num_patches
        head_dim = hidden_size // num_heads
        grid = int(round(num_patches ** 0.5))
        assert grid * grid == num_patches, "RoPE 目前只支持正方形 token grid"
        if self.rope:
            cos, sin = M.precompute_rope_2d(grid, head_dim, theta=self.rope_theta)
            self.register_buffer("rope_cos", cos.float(), persistent=False)
            self.register_buffer("rope_sin", sin.float(), persistent=False)
        else:
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)
        # 骨架辅助头（训练引导用，推理不用）：从 final_layer 前的 block 特征
        # 并行解出 1×32×32 latent 骨架预测，与 GT latent 骨架对齐。
        self.skel_head = None
        if self.skel_head_enabled:
            self.skel_head = nn.Sequential(
                M.build_norm(norm_type, hidden_size),
                nn.Linear(hidden_size, patch_size * patch_size, bias=True),
            )
        # 甲2 标准字形条件的 token-add 缩放(可学习, 初始 glyph_scale_init, 让字形条件有存在感)
        self.glyph_scale = nn.Parameter(torch.tensor(self.glyph_scale_init))
        # 甲2 标准字形条件：独立可训练 glyph_embedder(Conv2d 4→hidden, patch 编码)
        # 把标准字形 latent G(4,32,32) 编成与 x token 同形 token, forward 时注入。
        # 独立投影而非复用 x_embedder: 保证 g 编码可学习、norm 可控, 不依赖 x_embedder
        # 是否被冻结(XL LoRA 模式下 x_embedder 冻结, 复用会导致 g 信号被锁死)。
        self.glyph_embedder = None
        # 逐层注入（glyph_inject_layers>0 时启用）：见下方说明
        self.glyph_inject_layers = int(glyph_inject_layers)
        self.glyph_injections = None
        # 训练期随机丢弃标准字形条件的概率（见 forward 中注释）
        self.glyph_drop_prob = float(glyph_drop_prob)
        if self.use_glyph_cond:
            ps_ = self.x_embedder.patch_size[0] if not isinstance(self.x_embedder.patch_size, int) else self.x_embedder.patch_size
            self.glyph_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=ps_, stride=ps_, bias=False)
            #
            # 为什么需要逐层注入（这是本项目的关键设计修正）
            # ---------------------------------------------------------------
            # 原实现只在 **输入层** 做一次 token-add：
            #     x = x + glyph_scale * g_tok
            # 之后 x 要穿过 12 层 Transformer。每层 block 的输出都是
            # `x + block(x)`，会把 g 的相对贡献逐层稀释 —— 等于把答案写在
            # 第一页，然后让人翻完整本书再回答。
            #
            # 而 ControlNet 用的是 **逐层 zero-init adaLN 调制**（12 次），
            # 实测 SSIM 0.80。两者条件信息量完全相同（都是 4×32×32 空间图），
            # 差别只在注入方式。所以瓶颈不是「信息量不够」，而是「信息进不去」。
            #
            # 这里复用 controlnet.ZeroAdaLNInjection，与 ControlNet 对齐：
            #   out = x * (1 + s) + t，s/t 由 zero-init Linear 产出
            #   → init 时恒等，不破坏已有训练；且比加法注入表达力更强
            #     （既能增强也能抑制残差流）。
            if self.glyph_inject_layers > 0:
                from .controlnet import ZeroAdaLNInjection
                n_inj = min(self.glyph_inject_layers, depth)
                # 均匀分布在 depth 层中
                self.glyph_inject_at = sorted(
                    set(int(round((i + 1) * depth / n_inj)) - 1 for i in range(n_inj)))
                self.glyph_injections = nn.ModuleList([
                    ZeroAdaLNInjection(hidden_size, mode="modulate")
                    for _ in self.glyph_inject_at
                ])
        self.initialize_weights()
        if self.freeze_char_table and hasattr(self, "y_char_embedder"):
            # 冻结 char 表：DINO 预填充后不再训练（省 35130×384≈13.5M 训练参数），
            # 但保留最后一行的 CFG uncond 项可学习（它没有 DINO 对应物）。
            # 具体做法见 LabelEmbedder.freeze_table() —— 旧的
            # `w[-1].requires_grad_(True)` 是静默 no-op，null token 实际被冻结。
            _ye = self.y_char_embedder
            if hasattr(_ye, 'comp_embedding'):
                # IDSCharEmbedder: 冻结部件嵌入表
                _ye.comp_embedding.weight.requires_grad_(False)
                if _ye.null_embed is not None:
                    _ye.null_embed.requires_grad_(True)
                if _ye.fallback_embed is not None:
                    _ye.fallback_embed.requires_grad_(True)
            else:
                # LabelEmbedder: 冻结字符表
                with torch.no_grad():
                    w = _ye.embedding_table.weight
                    if w.shape[0] > 1:
                        w[-1].normal_(std=0.02)
                _ye.freeze_table()
            self._char_table_frozen = True
        else:
            self._char_table_frozen = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_callig_embedder.embedding_table.weight, std=0.02)
        # IDSCharEmbedder 用 comp_embedding 而非 embedding_table
        if hasattr(self.y_char_embedder, 'comp_embedding'):
            nn.init.normal_(self.y_char_embedder.comp_embedding.weight, std=0.02)
            if self.y_char_embedder.null_embed is not None:
                nn.init.normal_(self.y_char_embedder.null_embed, std=0.02)
            if self.y_char_embedder.fallback_embed is not None:
                nn.init.normal_(self.y_char_embedder.fallback_embed, std=0.02)
        elif hasattr(self.y_char_embedder, 'char_table'):
            # 标准字形 DINO 冻结表：不重新初始化（保持标准字形特征）。
            # 只初始化可学习的 CFG null token。
            if self.y_char_embedder.null_embed is not None:
                nn.init.normal_(self.y_char_embedder.null_embed, std=0.02)
        else:
            nn.init.normal_(self.y_char_embedder.embedding_table.weight, std=0.02)

        if getattr(self, "skel_head_enabled", False) and self.skel_head is not None:
            nn.init.zeros_(self.skel_head[-1].weight)
            nn.init.zeros_(self.skel_head[-1].bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # 标准字形的逐层注入必须**保持 zero-init**。
        #
        # ZeroAdaLNInjection 在构造时已把 proj 的 weight/bias 置零，但上面
        # 的 ``self.apply(_basic_init)`` 会把**所有** nn.Linear（含这些 proj）
        # 重新初始化成 xavier_uniform —— 于是注入不再是恒等映射，
        # warm-start 语义被破坏：从已训练 ckpt 续跑时，新加的注入层会给
        # 残差流注入随机扰动，等价于在训练好的模型上叠加噪声。
        #
        # 这里显式重新置零，恢复 ``out = x*(1+s)+t`` 中 s=t=0 的恒等起点。
        if getattr(self, "glyph_injections", None) is not None:
            for _inj in self.glyph_injections:
                nn.init.zeros_(_inj.proj.weight)
                nn.init.zeros_(_inj.proj.bias)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y_callig, y_char, return_intermediate_layer=None, g=None):
        """
        Forward pass of DiT_2Cond.
        x: (N, C, H, W) noisy latents
        t: (N,) timesteps
        y_callig, y_char: (N,) condition IDs
        g: (N, C, H, W) 标准字形 latent(与 x 同空间), 甲2 token-add 条件; None=不使用
        return_intermediate_layer: int block index (e.g. 8) whose patch features to return for REPA.
                                   When set, returns (output, intermediate_feats) as a tuple.
        """
        # ── 标准字形条件的随机丢弃（仅训练时）────────────────────────────
        #
        # 为什么要 drop：
        # 1. 防止「门控」。标准字形 latent 与 DINO CLS 都由字 ID 推出，存在冗余。
        #    std-skel 实验已证明：冗余条件会被 ControlNet 学成恒等门控掉
        #    （SSIM 0.494≈base，SkelIoU 0.015，12 个评测点全平）。
        #    随机丢弃使 g 不可预测，模型无法依赖「反正能从字 ID 推出来」而忽略它。
        # 2. 覆盖现实：fame 上 v2 字典只覆盖 53.1%，草/篆（46.9%）本来就没有
        #    标准字形，推理时 g 只能是零。训练时模拟这种缺失，让模型学会
        #    「g 有效时跟随，g 为零时依赖书家+字条件」。
        # 3. 与 CFG 兼容：丢弃时该样本等价于 uncond-g 分支。
        #
        # 注意：以**整张样本**为单位丢弃（而非逐像素），与「某字没有标准字形」
        # 的真实情况一致。
        if (g is not None and self.training and self.glyph_drop_prob > 0.0):
            N = g.shape[0]
            keep = torch.rand(N, device=g.device) >= self.glyph_drop_prob
            g = g * keep.view(N, 1, 1, 1).to(g.dtype)

        x = self.x_embedder(x)  # (N, T, D)
        g_tok = None            # 标准字形 token；未启用时保持 None（逐层注入会检查）
        if self.rope:
            # 位置信息由 RoPE 在 attention 内部注入，不再加到残差流上。
            # 这样 token 幅度不随位置编码偏移，也天然支持不同 grid 的外推。
            pass
        else:
            x = x + self.pos_embed
        if self.use_glyph_cond and self.glyph_embedder is not None and g is not None:
            # 独立 glyph_embedder 把标准字形 latent 编成 (N, D, 16, 16) -> flat tokens (N, 256, D)
            g_tok = self.glyph_embedder(g).flatten(2).transpose(1, 2)  # (N,256,D)
            x = x + self.glyph_scale * g_tok
        t_emb = self.t_embedder(t)               # (N, D)

        if self.condition_fusion == "factorized_add":
            # V3-A 二因子可组合 mask（4-way）：
            #   - drop_all           -> unconditional（CFG 基准）
            #   - drop_one & which=0 -> glyph-only（drop callig，学 content score s_G）
            #   - drop_one & which=1 -> callig-only（drop glyph，学 style score s_A）
            #   - 其余               -> full（callig+glyph，学 joint score）
            # 默认 0.10/0.30 配比 => full 60% / callig-only 15% / glyph-only 15% / uncond 10%。
            # 书家维度样本充足而字符维度才是难点: cond_drop_which_glyph_prob 让 drop-one
            # 偏向 glyph-only（drop callig 保 char），把专门训练预算给 5461 个字符内容分。
            if self.training and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0):
                r = torch.rand(y_callig.shape[0], device=y_callig.device)
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which_glyph = torch.rand(y_callig.shape[0], device=y_callig.device) < self.cond_drop_which_glyph_prob
                y_callig = torch.where(drop_all | (drop_one & which_glyph),
                                       self.y_callig_embedder.num_classes, y_callig)
                y_char = torch.where(drop_all | (drop_one & ~which_glyph),
                                     self.y_char_embedder.num_classes, y_char)
            e_callig = self.y_callig_embedder(y_callig, False)
            e_char = self.y_char_embedder(y_char, False)
            # 可学习幅度平衡：见 __init__ 处注释（DINO 区分度被书家分支淹没的实测）。
            y_emb = (self.callig_scale * self.callig_proj(e_callig)
                     + self.char_scale * self.char_proj(e_char)) / math.sqrt(2.0)
        elif self.condition_fusion == "xl_highdim":
            # XL 高维条件：与 factorized_add 相同的 4-way 可控 mask（CFG 需要 uncond 维度）。
            if self.training and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0):
                r = torch.rand(y_callig.shape[0], device=y_callig.device)
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which_glyph = torch.rand(y_callig.shape[0], device=y_callig.device) < self.cond_drop_which_glyph_prob
                y_callig = torch.where(drop_all | (drop_one & which_glyph),
                                       self.y_callig_embedder.num_classes, y_callig)
                y_char = torch.where(drop_all | (drop_one & ~which_glyph),
                                     self.y_char_embedder.num_classes, y_char)
            e_callig = self.y_callig_embedder(y_callig, False)
            e_char = self.y_char_embedder(y_char, False)
            y_emb = self.cond_fusion(torch.cat([e_callig, e_char], dim=-1)) * self.y_scale
        else:
            e_callig = self.y_callig_embedder(y_callig, self.training)
            e_char = self.y_char_embedder(y_char, self.training)
            y_concat = torch.cat([e_callig, e_char], dim=-1)
            y_emb = self.cond_fusion(y_concat)
        c = t_emb + y_emb                        # (N, D)

        rope = (self.rope_cos, self.rope_sin) if self.rope else None

        intermediate_feats = None
        # 逐层注入的层号 -> injection 模块索引
        _inj = {}
        if self.glyph_injections is not None and g_tok is not None:
            _inj = {blk: k for k, blk in enumerate(self.glyph_inject_at)}

        if self.use_checkpoint:
            for i, block in enumerate(self.blocks):
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    # Run this single block eagerly so its output can be captured for REPA.
                    x = block(x, c, rope=rope)
                    intermediate_feats = x
                else:
                    x = checkpoint(lambda *a: block(*a, rope=rope), x, c, use_reentrant=False)
                if i in _inj:
                    x = self.glyph_injections[_inj[i]](x, g_tok)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, c, rope=rope)
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    intermediate_feats = x
                if i in _inj:
                    # x = x*(1+s) + t，s/t 由 g_tok 经 zero-init Linear 产出。
                    # init 时恒等；梯度上 ∂out/∂g_tok = W = 0，因此这条路径
                    # 初期不给 glyph_embedder 梯度 —— 但输入层的
                    # x = x + glyph_scale * g_tok（glyph_scale=0.4 非零）
                    # 已提供直通梯度，故 glyph_embedder 从 step 0 即可学习。
                    x = self.glyph_injections[_inj[i]](x, g_tok)

        # 骨架头：从 final_layer 前的 block 输出特征并行解码 latent 骨架 (N,1,32,32)
        skel_pred = None
        if self.skel_head_enabled and self.skel_head is not None:
            skel_n = self.skel_head(x)                       # (N, T, p*p)，单通道 patch 值
            B_, T_, PP = skel_n.shape
            h_ = int(T_ ** 0.5)
            p_ = self.x_embedder.patch_size[0]
            # 完全镜像主 head 的 unpatchify（C=1）：
            # (B,H*W,p*p) -> (B,H,W,p,p,1) -> einsum 'nhwpqc->nchpwq' -> (B,1,H,W,p,p) -> (B,1,H*p,W*p)
            skel5 = skel_n.reshape(B_, h_, h_, p_, p_, 1)
            skel5 = torch.einsum('nhwpqc->nchpwq', skel5)
            skel_pred = skel5.reshape(B_, 1, h_ * p_, h_ * p_)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.skel_head_enabled and skel_pred is not None:
            # 返回 (主输出, skel_pred)；gaussian_diffusion.training_losses 会把第二元素
            # 当作 intermediate_feats 存入 loss_dict['intermediate_feats']
            return x, skel_pred
        if return_intermediate_layer is not None:
            return x, intermediate_feats
        return x

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, g=None):
        # Duplicate every sample: first copy conditional, second copy unconditional.
        original_bs = x.shape[0]
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y_callig = torch.cat([y_callig, y_callig], dim=0)
        y_char = torch.cat([y_char, y_char], dim=0)
        uncond_callig = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
        uncond_char = torch.full_like(y_char, self.y_char_embedder.num_classes)
        y_callig_combined = torch.cat([y_callig[:original_bs], uncond_callig[original_bs:]], dim=0)
        y_char_combined = torch.cat([y_char[:original_bs], uncond_char[original_bs:]], dim=0)
        # 标准字形条件 g 始终全给(两半都用真实 g): 字形内容是正条件, CFG 只强化 callig 风格
        g2 = torch.cat([g, g], dim=0) if g is not None else None
        model_out = self.forward(x, t, y_callig_combined, y_char_combined, g=g2)
        if isinstance(model_out, tuple):
            model_out = model_out[0]  # skel_head 启用时 forward 返回 (主输出, skel_pred)，CFG 只取主输出
        # Apply CFG on all learned channels (eps subspace), not a hard-coded prefix.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        out = torch.cat([eps, rest], dim=1)
        return out[:original_bs]

    def forward_with_2axis_cfg(self, x, t, y_callig, y_char,
                               cfg_callig=2.0, cfg_glyph=4.0, w_inter=0.0, g=None):
        """
        2-Axis Classifier-Free Guidance (style score + glyph content score + interaction score).
        Runs 4 parallel passes batched together along batch dimension:
          1. full     : (y_callig, y_char)        -> eps_full
          2. callig   : (y_callig, null_char)     -> eps_callig
          3. glyph    : (null_callig, y_char)     -> eps_glyph
          4. uncond   : (null_callig, null_char)  -> eps_uncond

        Möbius / Product-of-Experts composition:
          eps_guided = eps_uncond
                     + cfg_glyph  * (eps_glyph - eps_uncond)
                     + cfg_callig * (eps_callig - eps_uncond)
                     + w_inter    * (eps_full - eps_glyph - eps_callig + eps_uncond)
        """
        B = x.shape[0]
        x4 = torch.cat([x, x, x, x], dim=0)
        t4 = torch.cat([t, t, t, t], dim=0)

        null_c = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
        null_g = torch.full_like(y_char, self.y_char_embedder.num_classes)

        yc4 = torch.cat([y_callig, y_callig, null_c, null_c], dim=0)
        yg4 = torch.cat([y_char, null_g, y_char, null_g], dim=0)

        g4 = torch.cat([g, g, g, g], dim=0) if g is not None else None

        model_out = self.forward(x4, t4, yc4, yg4, g=g4)
        if isinstance(model_out, tuple):
            model_out = model_out[0]

        eps4, rest4 = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps_full, eps_callig, eps_glyph, eps_uncond = torch.split(eps4, B, dim=0)

        eps_guided = (
            eps_uncond
            + cfg_glyph * (eps_glyph - eps_uncond)
            + cfg_callig * (eps_callig - eps_uncond)
            + w_inter * (eps_full - eps_glyph - eps_callig + eps_uncond)
        )

        rest = rest4[:B]
        out = torch.cat([eps_guided, rest], dim=1)
        return out



def DiT_2Cond_XS_2(**kwargs):
    # 更小变体：depth=8, hidden=384, 6 头, patch=2。参数约 20M（-35% vs S/2 的 30M），
    # 适合小数据量（3top30 仅 3.8 万图）防过拟合；transformer 层从 12→8。
    return DiT_2Cond(depth=8, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_2Cond_WS_2(**kwargs):
    # 宽体变体：类别多时加宽 hidden 而非加深 depth。depth=8, hidden=768, 12 头。
    # 参数约 70M（2.3× S/2），patch=2 保持精细位置编码，适合类别区分任务。
    return DiT_2Cond(depth=8, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_2Cond_S_2(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_2Cond_S_4(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_2Cond_S_8(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_2Cond_B_2(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_2Cond_B_4(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


# 模型注册表。当前 pipeline 只用 **DiT-2Cond-S/2**（fame 预训练 + 1px ControlNet）。
#
# 2026-08-31 清理：移除了 'DiT-2Cond-XL/2'（从未在当前 pipeline 使用）。
# 一并删除的还有原版 DiT（单条件 + timm 组件）与 DiT_3Cond（三条件），
# 二者均已废弃；删掉后本文件不再依赖 timm。
DiT_2Cond_models = {
    'DiT-2Cond-XS/2': DiT_2Cond_XS_2,
    'DiT-2Cond-WS/2': DiT_2Cond_WS_2,
    'DiT-2Cond-S/2': DiT_2Cond_S_2,
    'DiT-2Cond-S/4': DiT_2Cond_S_4,
    'DiT-2Cond-S/8': DiT_2Cond_S_8,
    'DiT-2Cond-B/2': DiT_2Cond_B_2,
    'DiT-2Cond-B/4': DiT_2Cond_B_4,
}


