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


class MultiStyleEmbedder(nn.Module):
    """多模态风格查表 (v15)：每个风格类 (书家×书体 pair) K 个 token。

    ## 为什么需要（doc 72/73 + docs/919 §3）

    v13/v14 的书家条件 = 每类 **1 个** 128 维向量（SupCon 预训练 + 冻结）。
    样本多的书家风格多模态（苏轼 3131 张跨早晚期/多书体），单向量只能学"平均"：
    r(训练样本数, strict) = **−0.62**。v15 把每类扩成 K 个 token，让同一 pair 的
    K 个风格模态各占一个向量；token 由 DINO 特征 K-Means 质心初始化
    （tools/build_multistyle_k4.py），防止 K 个 token 训练初期隐式塌缩。

    ## 接口约定（与 LabelEmbedder 对齐）

    - ``embedding_table``: ``(num_classes, K*D)`` —— **不含** null 行；
    - ``null_embed``: 独立可学习 ``Parameter(K*D)``（对齐 freeze_table() 拆 null 的
      语义，但**从一开始就是独立参数**，不需要懒创建 —— materialize_lazy_params
      对本模块是 no-op）；
    - ``forward(labels, train)`` 返回 ``(B, K, D)``；``labels == num_classes``
      （4-way drop mask / CFG uncond 写入的 null 标签）整样本替换为 null_embed；
    - 条件 dropout **不由本模块驱动**（构造时 dropout_prob=0）：训练 forward 顶部的
      4-way drop mask 统一算好 null 标签后传入，保证 adaLN 分支与风格注入共享
      同一份 mask（双份独立 drop 会让两个分支看到不同的 uncond 样本）。

    adaLN 用的全局向量 = ``out.mean(dim=1)``（在 DiT_2Cond.forward 里做）。
    """

    def __init__(self, num_classes, k_clusters, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.k_clusters = int(k_clusters)
        self.hidden_size = int(hidden_size)
        self.dropout_prob = float(dropout_prob)
        self.embedding_table = nn.Embedding(self.num_classes,
                                            self.k_clusters * self.hidden_size)
        # CFG null token：独立可学习，形状对齐整行 (K*D)。std=0.02 同 LabelEmbedder。
        self.null_embed = nn.Parameter(
            torch.randn(self.k_clusters * self.hidden_size) * 0.02)

    def freeze_table(self):
        """冻结 [0, num_classes) 查表；null_embed 本就是独立 Parameter，保持可训。"""
        self.embedding_table.weight.requires_grad_(False)

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
        # null 标签 (== num_classes) 越界，先 clamp 到合法行再被 null_embed 整行覆盖
        out = self.embedding_table(labels.clamp(0, self.num_classes - 1))
        out = torch.where(null_mask.unsqueeze(-1),
                          self.null_embed.to(dtype=out.dtype), out)
        return out.view(-1, self.k_clusters, self.hidden_size)   # (B, K, D)


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















class ZeroCrossAttention(nn.Module):
    """零初始化 Cross-Attention 骨架注入 (GlyphDraw/IP-Adapter 式空间寻址).

    Q = x token (去噪画布, 内容已含位置信息), K/V = g_tok (标准字形特征)。
    与 ZeroAdaLNInjection (固定 1:1 位置对齐的逐 token 调制) 不同:
    每个 query token 动态聚合全部 N_ctx 个骨架 token (内容寻址), 2D 绑定
    靠 g 网格与 x 网格相同 (16×16) + 固定 sincos 位置嵌入保证。
    out_proj 零初始化 → 初始恒等, 不破坏已有训练。
    环境: cu121 torch>=2.0, 直接用 F.scaled_dot_product_attention。
    """

    def __init__(self, d_model, num_heads, grid_size=16, q_pos=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        grid_size = int(round(grid_size))
        # ⚠ `q_pos` (2026-09-17 新增, 默认 False = 旧行为, 保持 ckpt 兼容):
        #   原实现**只给 K/V 加位置, Q 没有**。而 rope=True 时 x 的残差流不加绝对位置
        #   (位置只进 attention 内部的 q/k), 所以 Q 是"无位置"的 ->
        #   所谓"空间寻址"退化成**内容寻址**: 每个 x token 只能靠内容相似度去猜该看哪个
        #   骨架 token, 无法可靠锁定同网格位置。这与本类 docstring 声称的
        #   "2D 绑定靠 g 网格与 x 网格相同 + 固定 sincos 位置嵌入保证"**不符**。
        #   打开 q_pos 后 Q 也加同一份 sincos, "2D 绑定"才真正成立。
        self.q_pos = bool(q_pos)
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # g 网格位置嵌入: 固定 sincos, 与 x 同 grid (N_ctx = grid_size²)
        pe = get_2d_sincos_pos_embed(d_model, grid_size)
        self.register_buffer(
            "ctx_pos", torch.from_numpy(pe).float().unsqueeze(0), persistent=False)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, context):
        return self._inject(x, context)

    def _inject(self, x, context):
        B, N, D = x.shape
        Nc = context.shape[1]
        if self.q_pos and N <= self.ctx_pos.shape[1]:
            q_in = x + self.ctx_pos[:, :N]
        else:
            q_in = x
        # K/V 位置: 前 grid² 个 token 是骨架 token(有空间语义, 加 sincos);
        # 超出网格的尾部 token (v15c 的风格 token) 是抽象语义, **不加位置**。
        # (原实现要求 Nc == grid², v15c 的 256+K context 在此崩溃。)
        if Nc <= self.ctx_pos.shape[1]:
            ctx_in = context + self.ctx_pos[:, :Nc]
        else:
            ctx_in = torch.cat(
                [context[:, :self.ctx_pos.shape[1]] + self.ctx_pos,
                 context[:, self.ctx_pos.shape[1]:]], dim=1)
        q = self.q_proj(self.norm_x(q_in)).view(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_c(ctx_in)) \
            .view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_c(ctx_in)) \
            .view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return x + self.out_proj(out)


class GlyphStyleCrossAttn(nn.Module):
    """每层注入: Q = x token, K/V = 外部拼好的 context = [骨架token(+pos); 风格token(+role)].

    与 ZeroCrossAttention 的关键区别: 本模块**不在内部加位置嵌入**
    (位置已在外部加到骨架 token, role 已加到风格 token), 因此风格 token 能
    **直接参与每一层**的内容寻址 —— 避免风格只经"骨架调制"间接进入后被深层稀释。

    每个 x 位置(query) 依据自身内容与空间位置, 同时聚合:
      - 局部字形  : 骨架 token(与 x 同 16x16 网格, 空间对应)
      - 书家风格  : style token(全局, 但 attention 权重随 query 位置变化
                    => 同一书家在不同位置吸收不同风格分量 = 风格局部化)

    这满足"图片上每一处都综合 局部字形 + 空间信息 + 书家风格 做调制"。
    out_proj 零初始化 -> 初始恒等, 可安全从已训 ckpt 续跑。
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, context):
        B, N, D = x.shape
        Nc = context.shape[1]
        q = self.q_proj(self.norm_x(x)).view(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = self.norm_c(context)
        k = self.k_proj(ctx).view(
            B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(
            B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return x + self.out_proj(out)


class CalligStyleCrossAttn(nn.Module):
    """书家化骨架 cross-attention（v15 多模态版）：Q = 骨架 token(+2D 位置), K/V = 风格 token。

    v1（单书家向量经内部 style_proj 展开成 n_style 个 token、Q 无位置编码）已
    存档到 ``legacy/dit_core/callig_style_cross_attn_v1.py``——没有任何已训练
    run 用过它（doc 60），删除不破坏任何 ckpt 复评。

    v15 数据流：MultiStyleEmbedder 查表得到 ``(B, K, D)`` 风格 token
    （DINO K-Means 质心初始化，每 token 对应该 pair 的一个风格模态），本模块
    让每个骨架位置依自身内容+空间位置在 K 个模态间做内容寻址聚合。
    Q 加 2D sincos 位置：骨架网格与画布同为 16×16，位置对齐让"空间寻址"成立
    （ZeroCrossAttention 的 q_pos 教训——Q 无位置时退化为纯内容寻址）。
    out_proj zero-init：resume 起点恒等，step0 输出等于原 ckpt。
    """

    def __init__(self, hidden_size, num_heads, grid_size=16):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # 骨架空间位置编码：固定 sincos（16×16 网格），Q 专用；K/V 是抽象风格
        # token（无空间语义），不加位置。
        pe = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.register_buffer(
            "ctx_pos", torch.from_numpy(pe).float().unsqueeze(0), persistent=False)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, g_tok, style_tokens):
        """g_tok: (B, 256, D) 骨架 token; style_tokens: (B, K, D) -> 书家化骨架."""
        B, Nq, D = g_tok.shape
        Nk = style_tokens.shape[1]
        q_in = g_tok + self.ctx_pos[:, :Nq]
        q = self.q_proj(self.norm_q(q_in)).view(
            B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(style_tokens)).view(
            B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(style_tokens)).view(
            B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return g_tok + self.out_proj(out)


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
        # ---- g(标准字形) 的**向量**因子 (v12) ----
        # 让 g 以"全局内容向量"的身份进入条件向量 c, 而不是只走 token-add / 逐层注入。
        #
        # 动机 (两个):
        #  1) ref(Moyun) 的条件融合是 cat([callig, font, char]) -> Linear(3h->h),
        #     即**特征维拼接多个向量因子**。我们只有一个向量因子(callig), 直接照搬
        #     会退化成单层 Linear (与 factorized_add 等价, 参数量实测完全相同)。
        #     把 g 池化成向量作为第二个操作数, concat 才真正非退化。
        #  2) 补一个真实的洞: 现在 c = t_emb + callig_proj(e_callig), **adaLN 调制
        #     分支从来看不到内容** —— 每个 block 的全局 scale/shift/gate 都不知道
        #     在写哪个字。g 只经 token-add(输入层) 与 ZeroAdaLNInjection(4 层) 进入。
        #
        # 两种融合方式共用同一组操作数 {e_callig, e_glyph_vec}, 便于做纯融合方式的对照:
        #   factorized_add: 各自投影到 hidden 后按可学习标量相加
        #   factorized_cat: 拼接后经一个联合 Linear (ref 式)
        glyph_vec_cond=False,
        glyph_vec_dim=128,
        glyph_vec_pool="mean",   # "mean" | "max"  (对 g_tok 的 256 个 token 做池化)
        cond_drop_all_prob=0.05,
        cond_drop_one_prob=0.0,
        cond_drop_which_glyph_prob=0.5,
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=0.4,
        # v10b: 去掉 char 向量条件 (skel-g 即字条件时的干净因子分解: 结构=skel, 风格=callig)。
        # False 时不建 y_char_embedder/char_proj/char_scale, 4-way 退化为 callig 单向量因子
        # (drop_all/drop_one 都丢 callig); forward 仍接受 y_char 形参但忽略 (接口零破坏)。
        use_char_cond=True,
        # 标准字形条件的逐层注入层数。0 = 关闭（只用输入层 token-add，旧行为）；
        # >0 = 在该数量的 block 后注入（均匀分布），与 ControlNet 的
        # ZeroAdaLNInjection 完全对齐。见下方 glyph_embedder 处的说明。
        # 显存：每层约 +150MB（batch=192 时），12 层约 +1.8G，注意 OOM。
        glyph_inject_layers=0,
        glyph_inject_mode="adaln",
        # [2026-09-17] xattn 的 Q 是否也加 sincos 位置嵌入。默认 False = 旧行为。
        # 旧实现只给 K/V 加位置, 而 rope=True 时 x 残差流不加绝对位置 -> Q 无位置,
        # "空间寻址"退化成"内容寻址"。打开后 Q/K/V 都带位置, 2D 绑定才成立。
        xattn_q_pos=False,
        glyph_in_channels=4,   # g 骨架 latent 的通道数 (aux 目标通道不改变它)
        # 标准字形条件的训练期随机丢弃概率。0 = 不丢弃。
        # 作用见 forward() 中的注释：防门控 + 模拟草/篆无标准字形的真实缺失。
        glyph_drop_prob=0.0,
        # 标准字形编码器深度。0 = 单层 Conv2d(现状)；>0 = 降采样 Conv +
        # depth 层 Conv+SiLU(逐层) 增强 std 骨架 latent 的编码能力。
        # 动机(fame3 诊断)：std-g 下 g 注入作用仅 ~2.7%(GT-g 10.4%)、glyph_scale
        # 梯度近零 —— 除 glyph_drop 外, 单层编码器可能提取不出足够结构特征。
        glyph_embedder_depth=0,
        # [v12+] glyph_embedder 的 3x3 卷积用 **depthwise-separable** 实现。
        #
        # 动机 (FLOP 实测, _review/flop_profile.py):
        #   glyph_embedder_depth=2 的两层满秩 3x3 conv 在 h=384 上各需
        #   16*16*384*384*9 = 339.7M MACs, 合计 680M = **全模型 9.7% 的 FLOPs**。
        #   而 depthwise-separable 同感受野只需 384*9 + 384*384 = 150.9K/位置
        #   -> 38.6M, 即 **8.8x 更便宜** (节省约 8.6% 总 FLOPs)。
        #   标准 MobileNet 式分解, 感受野/表达能力基本等价。
        # 默认 False = 保持原满秩实现, 不改变任何已有 ckpt 的行为。
        glyph_embedder_sep=False,
        char_proj_mode="full",
        callig_proj_mode="linear",   # 42 号实验: "mlp" 两层 MLP 补 callig 容量
        callig_scale_init=1.0,       # 42 号实验: callig_scale 初值 (1.5 增强风格权重)
        # ---- 多模态风格 (v15): 每类 K 个 style token + 三种注入方式矩阵 ----
        # >0 时 y_callig_embedder 换成 MultiStyleEmbedder（查表 (B,K,D)，DINO
        # K-Means 质心初始化），token dim = callig_embed_dim（None 则 = hidden）。
        # 注入方式（从头训练矩阵 v15a/b/c，单变量对照）:
        #   a) 仅 mean pooling -> adaLN（callig_style_ca=false, style_ctx=false）
        #   b) + CalligStyleCrossAttn 书家化骨架（callig_style_ca=true）
        #   c) + K token 拼进每层 xattn context（style_ctx_every_layer=true,
        #      需 glyph_inject_mode=xattn）
        callig_multi_style_k=0,
        callig_style_ca=False,
        style_ctx_every_layer=False,
        # ---- 外挂 callig_spatial (已证伪死重, 保留为可配置开关以复评历史 ckpt) ----
        callig_spatial=False,
        callig_spatial_rank=64,
        # ---- 风格 token 直接参与每层注入 (GlyphStyleCrossAttn) ----
        # style_token_n>0 时: 注入 context = [书家化骨架(+pos); 风格token(+role)],
        # 使风格在**每一层**都可见, 而非只经骨架间接进入被深层稀释。
        # 这是"图片每一处综合 局部字形+空间+风格 调制"的载体。
        style_token_n=0,
        style_role_init=0.02,
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
        image_channels=None,    # CFG 只作用于 image latent 通道; None 时退回 in_channels (向后兼容)
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.image_channels = int(image_channels) if image_channels is not None else in_channels
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
        self.use_char_cond = bool(use_char_cond)
        self.char_proj_mode = char_proj_mode
        self.freeze_char_table = bool(freeze_char_table)
        if self.cond_drop_all_prob < 0 or self.cond_drop_one_prob < 0:
            raise ValueError("condition dropout probabilities must be non-negative")
        if self.cond_drop_all_prob + self.cond_drop_one_prob > 1:
            raise ValueError("cond_drop_all_prob + cond_drop_one_prob must be <= 1")

        # 用 modules 版（自带 RMSNorm/SwiGLU/RoPE/QK-Norm），不再依赖 timm。
        self.x_embedder = M.PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # g 向量因子: 仅在 factorized_add / factorized_cat 下可用。先给默认值,
        # 保证 legacy / xl_highdim 分支下这些属性也存在 (forward 里按 glyph_vec_cond 守卫)。
        self.glyph_vec_cond = bool(glyph_vec_cond)
        self.glyph_vec_pool = str(glyph_vec_pool)
        self.glyph_vec_proj = None
        self.glyph_vec_out = None
        self.glyph_vec_scale = None

        # ⚠ 这些属性此前**只在 factorized_add 分支里赋值**, 而 forward() 在
        #   `use_glyph_cond and g is not None` 时会无条件访问 self.callig_style_ca
        #   → condition_fusion="legacy"(默认值!) + w_glyph_cond=1 会直接 AttributeError。
        #   这里统一给默认值, 使任何 fusion 组合都安全。
        self.callig_style_ca = None
        self.callig_basis = None
        self.callig_spatial_net = None
        self.style_proj = None
        self.style_role = None
        self.ctx_pos_g = None
        # forward 的多模态路由读它; 非 factorized 分支恒为 0
        self.callig_multi_style_k = int(callig_multi_style_k)
        self.style_ctx_every_layer = False

        if condition_fusion in ("factorized_add", "factorized_cat"):
            # 二因子可组合条件（V3-A）：calligrapher（风格）× glyph（内容=script×char 合并类）。
            # 每个因子独立低维 embedding。
            #
            # 两种融合方式（v12 起可选，见下方 cond_fusion 处说明）：
            #   factorized_add: 各因子**独立投影**后按可学习标量相加
            #   factorized_cat: 各因子 embedding **拼接**后经一个联合 Linear 投影（ref/Moyun 式）
            # 未见的 (callig, glyph) 组合
            # 由两个各自训练充分的边际 score 组合而成，而不是靠一整张联合表 memorization。
            callig_embed_dim = callig_embed_dim or hidden_size
            char_embed_dim = char_embed_dim or hidden_size
            # v15 多模态风格: 每类 K 个 token 的查表 (动机见 MultiStyleEmbedder);
            # 否则维持单向量 LabelEmbedder (v13/v14 口径, 含 null 行 + 懒 null_embed)。
            if self.callig_multi_style_k > 0:
                if callig_spatial:
                    raise ValueError(
                        "callig_multi_style_k 与 callig_spatial 互斥 "
                        "(外挂基图路径读 2-D e_callig, 多模态查表输出 (B,K,D))")
                if int(style_token_n) > 0:
                    raise ValueError(
                        "callig_multi_style_k 与 style_token_n>0 互斥 "
                        "(每层注入的 style_proj 读 2-D e_callig)")
                self.y_callig_embedder = MultiStyleEmbedder(
                    num_calligraphers, self.callig_multi_style_k,
                    callig_embed_dim, dropout_prob=0.0)
            else:
                self.y_callig_embedder = LabelEmbedder(
                    num_calligraphers, callig_embed_dim, 0.0, use_cfg_embedding=True)
            if not self.use_char_cond:
                # v10b: skel-g 即字条件, char 向量因子整体移除
                self.y_char_embedder = None
                self.char_proj = None
                self.char_scale = None
            elif use_std_dino_char_embedder:
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
            if callig_proj_mode == "mlp":
                # 42 号实验: callig 链增强 —— 两层 MLP 补容量 (诊断: callig_chain
                # rel 0.0013 弱梯度, 128 维单层容量不足)。
                self.callig_proj = nn.Sequential(
                    nn.LayerNorm(callig_embed_dim),
                    nn.Linear(callig_embed_dim, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            else:
                self.callig_proj = nn.Sequential(nn.LayerNorm(callig_embed_dim),
                                                 nn.Linear(callig_embed_dim, hidden_size))
            if not self.use_char_cond:
                pass    # char 侧已在上方整体移除
            elif char_proj_mode == "ln_only":
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
            self.callig_scale = nn.Parameter(torch.tensor(float(callig_scale_init)))
            if self.use_char_cond:
                self.char_scale = nn.Parameter(torch.tensor(1.0))
            # ---- g 的向量因子: 池化 g_tok -> 低维向量 (见 __init__ 参数处说明) ----
            # 与 callig 一起构成 concat 的两个操作数 (或 add 的两个加数)。
            if self.glyph_vec_cond:
                # g_tok 的 256 个 token 各是 hidden 维 -> 池化后投影到 glyph_vec_dim
                self.glyph_vec_proj = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, int(glyph_vec_dim)),
                )
            else:
                self.glyph_vec_proj = None
            if condition_fusion == "factorized_cat":
                # ---- concat 融合 (ref/Moyun LabelEmbedder 式, v12) ----
                # 所有向量因子的 embedding **拼接**后, 经**一个**联合 Linear 投影到 hidden。
                #
                # 与 factorized_add 的区别:
                #   add: 各因子独立投影到 hidden 后相加 -> 因子间交互只能靠后续 adaLN 间接产生
                #   cat: Linear 一次性看到全部因子 -> 能直接建模因子间交互,
                #        表达力严格更强 (ref 即此做法: concat(3张表) -> Linear(3h -> h))
                #
                # 操作数集合 (两者一致, 便于做纯融合方式的对照):
                #   e_callig (callig_embed_dim)  [+ e_char 若 use_char_cond]
                #   e_glyph_vec (glyph_vec_dim)  若 glyph_vec_cond
                # ⚠ 若两个操作数都没有(callig 之外无因子且 glyph_vec_cond=False),
                #   cat 会退化成单层 Linear, 与 factorized_add 等价 —— 参数量完全相同。
                _cat_dim = callig_embed_dim + (char_embed_dim if self.use_char_cond else 0)
                if self.glyph_vec_cond:
                    _cat_dim += int(glyph_vec_dim)
                self.cond_fusion = nn.Sequential(
                    nn.LayerNorm(_cat_dim),
                    nn.Linear(_cat_dim, hidden_size),
                )
                # 置 None 而非保留: 避免 DDP 出现"未参与前向的参数"报错
                # (callig_proj/char_proj/scale 在 cat 模式下不再使用)。
                self.callig_proj = None
                self.char_proj = None
                self.callig_scale = None
                self.char_scale = None
            else:
                self.cond_fusion = None
                if self.glyph_vec_cond:
                    # add 模式: glyph 向量单独投影到 hidden, 配一个可学习标量,
                    # 与 callig 分支对称 (两者操作数集合与 cat 模式完全相同)。
                    self.glyph_vec_out = nn.Sequential(
                        nn.LayerNorm(int(glyph_vec_dim)),
                        nn.Linear(int(glyph_vec_dim), hidden_size),
                    )
                    self.glyph_vec_scale = nn.Parameter(torch.tensor(1.0))
                else:
                    self.glyph_vec_out = None
                    self.glyph_vec_scale = None
            # callig 风格 cross-attention (v15b): 多模态风格 token 调制骨架
            # (见 CalligStyleCrossAttn)。zero-init -> step0 恒等。
            if self.callig_multi_style_k > 0 and callig_style_ca:
                self.callig_style_ca = CalligStyleCrossAttn(
                    hidden_size, num_heads=num_heads,
                    grid_size=int(self.x_embedder.num_patches ** 0.5))
            else:
                self.callig_style_ca = None
            # 外挂(callig_spatial, 低秩): 书家向量 -> r 个系数, 线性组合 r 张可学习空间基图
            # (r,256,D) -> (N,256,D), 逐 token 加到骨架 g_tok 上。实测 strict ±0.002 (死重),
            # 但保留为可配置开关以支持历史 ckpt 复评 (cos_e 等)。
            if callig_spatial:
                _n_patch = int(self.x_embedder.num_patches)
                _r = int(callig_spatial_rank)
                self.callig_basis = nn.Parameter(torch.empty(_r, _n_patch, hidden_size))
                nn.init.normal_(self.callig_basis, std=0.02)
                self.callig_spatial_net = nn.Sequential(
                    nn.LayerNorm(callig_embed_dim),
                    nn.Linear(callig_embed_dim, _r),
                )
            else:
                self.callig_basis = None
                self.callig_spatial_net = None
            # 风格 token(每层注入可见): 共享投影, 各层复用同一组 style token。
            # role 可学习 -> 给 N 个 token 不同"角色", 防止塌缩为同一向量。
            self.n_style_token = int(style_token_n)
            # v15c: K 个多模态风格 token 拼进**每层**注入 context (K/V 侧可见)。
            # 与 style_token_n>0 的旧路径互斥 (那条路用 style_proj 从 2-D e_callig
            # 造 token; 这里的 token 直接来自 MultiStyleEmbedder, role 只是加性偏置)。
            self.style_ctx_every_layer = bool(style_ctx_every_layer)
            if self.style_ctx_every_layer:
                if self.callig_multi_style_k == 0:
                    raise ValueError("style_ctx_every_layer 需要 callig_multi_style_k > 0")
                if self.n_style_token > 0:
                    raise ValueError("style_ctx_every_layer 与 style_token_n>0 互斥")
                if str(glyph_inject_mode) != "xattn" or int(glyph_inject_layers) == 0:
                    raise ValueError(
                        "style_ctx_every_layer 需要 glyph_inject_mode='xattn' 且 "
                        "glyph_inject_layers>0 (adaln 注入按位置对齐消费 context, "
                        "多出来的 K 个 token 会被忽略)")
                self.style_role = nn.Parameter(
                    torch.randn(self.callig_multi_style_k, hidden_size)
                    * float(style_role_init))
                _pe = get_2d_sincos_pos_embed(
                    hidden_size, int(self.x_embedder.num_patches ** 0.5))
                if "ctx_pos_g" in self.__dict__:
                    del self.__dict__["ctx_pos_g"]
                self.register_buffer(
                    "ctx_pos_g", torch.from_numpy(_pe).float().unsqueeze(0),
                    persistent=False)
            # 风格 token(每层注入可见): 共享投影, 各层复用同一组 style token。
            # role 可学习 -> 给 N 个 token 不同"角色", 防止塌缩为同一向量。
            if self.n_style_token > 0:
                self.style_proj = nn.Sequential(
                    nn.LayerNorm(callig_embed_dim),
                    nn.Linear(callig_embed_dim, self.n_style_token * hidden_size))
                self.style_role = nn.Parameter(
                    torch.randn(self.n_style_token, hidden_size)
                    * float(style_role_init))
                _pe = get_2d_sincos_pos_embed(hidden_size, 16)
                # ⚠ 上面"默认安全值"那段已经把 ctx_pos_g 设成了普通属性 (None)，
                #   而 register_buffer 对**已存在的属性名**会抛
                #   KeyError: "attribute 'ctx_pos_g' already exists"。
                #   必须先删掉再注册。（实测踩到：说明 xattn + style_token_n>0
                #   这条路径历史上从未成功跑过 —— 所有 run 都是 adaln + style_token_n=0）
                if "ctx_pos_g" in self.__dict__:
                    del self.__dict__["ctx_pos_g"]
                self.register_buffer(
                    "ctx_pos_g", torch.from_numpy(_pe).float().unsqueeze(0),
                    persistent=False)
            else:
                self.style_proj = None
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
        self.glyph_inject_mode = str(glyph_inject_mode)
        self.xattn_q_pos = bool(xattn_q_pos)
        self.glyph_injections = None
        # "block 下标 -> 注入器下标" 映射（glyph_inject_layers>0 时在下面填充）。
        # 放在 forward 外面是因为它是**循环不变量**，原来每步重建一个 dict。
        self._inj_map = {}
        # 训练期随机丢弃标准字形条件的概率（见 forward 中注释）
        self.glyph_drop_prob = float(glyph_drop_prob)
        self.glyph_embedder_depth = int(glyph_embedder_depth)
        self.glyph_embedder_sep = bool(glyph_embedder_sep)
        if self.use_glyph_cond:
            ps_ = self.x_embedder.patch_size[0] if not isinstance(self.x_embedder.patch_size, int) else self.x_embedder.patch_size
            if self.glyph_embedder_depth <= 0:
                self.glyph_embedder = nn.Conv2d(glyph_in_channels, hidden_size, kernel_size=ps_, stride=ps_, bias=False)
            elif self.glyph_embedder_sep:
                # [v12+] depthwise-separable 版: 同感受野/同通道数, 但把空间混合与
                # 通道混合拆开 -> 3x3 那层的代价从 h*h*9 降到 h*9 + h*h (约 8.8x)。
                # 见 __init__ 参数 glyph_embedder_sep 处说明与 FLOP 实测。
                _layers = [nn.Conv2d(glyph_in_channels, hidden_size,
                                     kernel_size=ps_, stride=ps_, bias=False)]
                for _ in range(self.glyph_embedder_depth):
                    _layers.append(nn.SiLU())
                    _layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                             stride=1, padding=1, bias=False,
                                             groups=hidden_size))          # depthwise
                    _layers.append(nn.Conv2d(hidden_size, hidden_size,
                                             kernel_size=1, bias=False))   # pointwise
                self.glyph_embedder = nn.Sequential(*_layers)
            else:
                # 增强编码器: 降采样 Conv + depth 层 SiLU+Conv(3x3 保分辨率),
                # 提升 std 骨架 latent 的结构特征提取能力(fame3 诊断: g 注入作用仅 ~2.7%)。
                _layers = [nn.Conv2d(glyph_in_channels, hidden_size, kernel_size=ps_, stride=ps_, bias=False)]
                for _ in range(self.glyph_embedder_depth):
                    _layers.append(nn.SiLU())
                    _layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=False))
                self.glyph_embedder = nn.Sequential(*_layers)
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
                n_inj = min(self.glyph_inject_layers, depth)
                # 均匀分布在 depth 层中
                self.glyph_inject_at = sorted(
                    set(int(round((i + 1) * depth / n_inj)) - 1 for i in range(n_inj)))
                # ★ 2026-09-17: 预计算 "block 下标 -> 注入器下标" 映射。
                #   原来在 forward 里每步重建这个 dict（`{blk: k for ...}`）——
                #   它是**循环不变量**（只依赖 glyph_inject_at），没必要每步做。
                self._inj_map = {blk: k for k, blk in enumerate(self.glyph_inject_at)}
                # glyph_inject_mode: "adaln" = ZeroAdaLNInjection (固定 1:1 位置调制,
                # 旧 ckpt 兼容默认); "xattn" = ZeroCrossAttention (空间寻址注入,
                # GlyphDraw/IP-Adapter 式内容+风格解耦, 2026-09-08)
                if glyph_inject_mode == "xattn":
                    if self.n_style_token > 0:
                        # 风格 token 每层可见: context 由 forward 外部拼好
                        # (骨架+pos ; 风格+role), 本类不再内部加位置。
                        self.glyph_injections = nn.ModuleList([
                            GlyphStyleCrossAttn(hidden_size, num_heads=num_heads)
                            for _ in self.glyph_inject_at
                        ])
                    else:
                        self.glyph_injections = nn.ModuleList([
                            ZeroCrossAttention(hidden_size, num_heads=num_heads,
                                               grid_size=self.x_embedder.num_patches ** 0.5,
                                               q_pos=bool(xattn_q_pos))
                            for _ in self.glyph_inject_at
                        ])
                else:
                    # ZeroAdaLNInjection 现居 legacy/controlnet.py (ControlNet 线归档时
                    # 迁入), 但它仍是 adaln 注入的**活跃实现** (v13/v14/v15 全在用)。
                    from .legacy.controlnet import ZeroAdaLNInjection
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
            elif hasattr(_ye, 'char_table'):
                # StdDinoCharEmbedder: 冻结 buffer 查表, 天然不参与梯度;
                # 仅确保 CFG null token 可学习 (与 IDS 分支一致)。
                if _ye.null_embed is not None:
                    _ye.null_embed.requires_grad_(True)
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
        # v10b: use_char_cond=False 时 char 侧整体不存在
        # IDSCharEmbedder 用 comp_embedding 而非 embedding_table
        if self.y_char_embedder is None:
            pass    # v10b: 无 char 侧
        elif self.y_char_embedder is not None and hasattr(self.y_char_embedder, 'comp_embedding'):
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
                if hasattr(_inj, "out_proj"):      # ZeroCrossAttention
                    nn.init.zeros_(_inj.out_proj.weight)
                    nn.init.zeros_(_inj.out_proj.bias)
                else:                              # ZeroAdaLNInjection
                    nn.init.zeros_(_inj.proj.weight)
                    nn.init.zeros_(_inj.proj.bias)

        # callig 空间化最后一层 (to_spatial) 同样恢复 zero-init:
        # s_callig=0 初始恒等, 从已训 ckpt 续跑时不注入随机扰动。
        if getattr(self, "callig_spatial_net", None) is not None:
            nn.init.zeros_(self.callig_spatial_net[-1].weight)
            nn.init.zeros_(self.callig_spatial_net[-1].bias)

        # callig 风格 cross-attn 的 out_proj 同样恢复 zero-init (resume 恒等)。
        if getattr(self, "callig_style_ca", None) is not None:
            nn.init.zeros_(self.callig_style_ca.out_proj.weight)
            nn.init.zeros_(self.callig_style_ca.out_proj.bias)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y_callig, y_char, return_intermediate_layer=None,
                return_intermediate_layers=None, g=None):
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
        else:
            keep = None

        # ── callig 条件 drop mask: 在 g 路径**之前**计算并全程共享 ────────────
        # (adaLN 分支与骨架风格注入必须用同一份 drop mask, 否则 drop-callig 样本
        #  的 adaLN 是 null 向量而骨架通路仍带真实风格 -> uncond 分支被污染)
        callig_drop = None
        char_drop = None
        # ⚠ 这个元组必须与**所有会读 `y_callig_in` / `char_drop` 的融合分支**一一对应。
        #   2026-09-17 修: 原先漏了 "factorized_cat" —— 该分支在下方同样复用
        #   `y_callig_in` / `char_drop`, 但因不在元组里, 整个 drop 块被跳过 ->
        #   `callig_drop is None` -> `y_callig_in = y_callig` -> **训练全程零条件 dropout**,
        #   null token 从未出现, `null_embed` 停在随机初始化。
        #   后果: 推理时 `forward_with_cfg` 的 uncond 半用的是**未训练**的 null 向量,
        #   `cfg=0.7` 实际是在往一个随机方向插值, 而不是真正的 CFG。
        #   不报错、loss 正常下降 —— 又一例静默失效。**新增 fusion 分支时必须回来加。**
        if (self.condition_fusion in ("factorized_add", "factorized_cat", "xl_highdim")
                and self.training
                and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0)):
            r = torch.rand(y_callig.shape[0], device=y_callig.device)
            if not self.use_char_cond:
                # v10b 单向量因子: drop_all/drop_one 同义
                callig_drop = r < (self.cond_drop_all_prob + self.cond_drop_one_prob)
            else:
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which_glyph = torch.rand(y_callig.shape[0], device=y_callig.device) < self.cond_drop_which_glyph_prob
                callig_drop = drop_all | (drop_one & which_glyph)
                char_drop = drop_all | (drop_one & ~which_glyph)
        y_callig_in = (torch.where(callig_drop, self.y_callig_embedder.num_classes, y_callig)
                       if callig_drop is not None else y_callig)

        # ── v15 多模态风格 token：一次查表, 骨架注入与 adaLN 融合共享 ──────────
        # (B, K, D); drop 已由 y_callig_in 携带 (null 标签 -> null_embed 整组替换)
        style_tokens = None
        if self.callig_multi_style_k > 0:
            style_tokens = self.y_callig_embedder(y_callig_in, False)

        def _e_callig():
            """adaLN 融合的书家向量因子：多模态 = K token mean pooling, 其余 = 查表。"""
            if style_tokens is not None:
                return style_tokens.mean(dim=1)      # (N, D)
            return self.y_callig_embedder(y_callig_in, False)

        x = self.x_embedder(x)  # (N, T, D)
        g_tok = None            # 标准字形 token；未启用时保持 None（逐层注入会检查）
        if self.rope:
            # 位置信息由 RoPE 在 attention 内部注入，不再加到残差流上。
            # 这样 token 幅度不随位置编码偏移，也天然支持不同 grid 的外推。
            pass
        else:
            x = x + self.pos_embed
        if self.use_glyph_cond and self.glyph_embedder is not None and g is not None:
            # 独立 glyph_embedder 把标准字形 latent 编成 (N, D, 16, 16) -> flat tokens (N,256,D)
            g_tok = self.glyph_embedder(g).flatten(2).transpose(1, 2)  # (N,256,D)
            if self.callig_style_ca is not None:
                # 书家化骨架(cross-attn): 骨架 token(+2D 位置) 在 K 个风格模态间
                # 内容寻址聚合, 产生"书家x字x位置"交互(结体差异)。
                # 注意 out_proj zero-init -> 注入从 0 平滑启动, 残差注入标准做法。
                # style_tokens 用 drop 后的 y_callig_in 查表: drop-callig 样本的
                # style token 全组来自 null_embed, 与 adaLN 分支一致。
                g_tok = self.callig_style_ca(g_tok, style_tokens)
            if getattr(self, "callig_spatial_net", None) is not None and self.callig_basis is not None:
                # 外挂(低秩): 书家向量 -> r 系数 x (r,256,D) 基图 -> 逐 token 加到骨架
                e_c_sp = self.y_callig_embedder(y_callig_in, False)
                _coef = self.callig_spatial_net(e_c_sp)                    # (N, r)
                g_tok = g_tok + torch.einsum("nr,rpd->npd", _coef, self.callig_basis)
            if keep is not None:
                # ⚠ 丢弃语义保护: style 注入会给零骨架加非零风格输出, 会把
                # "uncond-g 分支"(drop 的样本)重新变成有条件 —— 必须**在风格调制
                # 之后**把被丢弃样本的 g_tok 重新置零, 保持 drop 分支纯净。
                g_tok = g_tok * keep.view(-1, 1, 1).to(g_tok.dtype)
            x = x + self.glyph_scale * g_tok

        # ── g 的全局内容向量 (进入条件向量 c) ────────────────────────────────
        # 把 g_tok 的 256 个 token 池化成一个向量, 作为 concat/add 的第二个操作数。
        # 池化源用**已做 drop 掩码后**的 g_tok, 保证 drop-g 样本的该向量也为零
        # (与"该样本没有 g 条件"的语义一致)。
        e_glyph_vec = None
        if self.glyph_vec_cond:
            if g_tok is not None:
                _pooled = (g_tok.amax(dim=1) if self.glyph_vec_pool == "max"
                           else g_tok.mean(dim=1))
            else:
                # 完全没有 g 时走零向量 —— 仍经过 proj, 保证 DDP 下该分支参数
                # 参与前向 (否则报 "parameters that didn't receive grad")。
                _pooled = torch.zeros(x.shape[0], self.x_embedder.hidden_size,
                                      device=x.device, dtype=x.dtype)
            e_glyph_vec = self.glyph_vec_proj(_pooled)

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
            # 注意: drop mask 已在 forward 顶部计算 (y_callig_in 与骨架风格注入共享),
            # 这里不再重复 roll (重复 roll 会导致两分支 mask 不一致)。
            if not self.use_char_cond:
                # v10b: 单向量因子 (callig), drop_all/drop_one 同义 —— 丢 callig = uncond 向量
                e_callig = _e_callig()
                y_emb = self.callig_scale * self.callig_proj(e_callig)
            else:
                if self.training and char_drop is not None:
                    y_char = torch.where(char_drop, self.y_char_embedder.num_classes, y_char)
                e_callig = _e_callig()
                e_char = self.y_char_embedder(y_char, False)
                # 可学习幅度平衡：见 __init__ 处注释（DINO 区分度被书家分支淹没的实测）。
                y_emb = (self.callig_scale * self.callig_proj(e_callig)
                         + self.char_scale * self.char_proj(e_char)) / math.sqrt(2.0)
            if self.glyph_vec_cond and e_glyph_vec is not None:
                # g 向量因子: 与 callig 分支对称的"独立投影 + 可学习标量"加数。
                # 操作数集合与 factorized_cat 完全一致 -> 两种融合方式可直接对照。
                y_emb = y_emb + self.glyph_vec_scale * self.glyph_vec_out(e_glyph_vec)
        elif self.condition_fusion == "factorized_cat":
            # ref(Moyun) 式 concat 融合: 各向量因子 embedding 拼接 -> 联合 LN+Linear。
            # 见 __init__ 处说明。操作数 = {e_callig [, e_char] [, e_glyph_vec]}。
            # drop mask 同样复用 forward 顶部算好的那份 (y_callig_in / char_drop)。
            e_callig = _e_callig()
            _parts = [e_callig]
            if self.use_char_cond:
                if self.training and char_drop is not None:
                    y_char = torch.where(char_drop, self.y_char_embedder.num_classes, y_char)
                _parts.append(self.y_char_embedder(y_char, False))
            if self.glyph_vec_cond and e_glyph_vec is not None:
                _parts.append(e_glyph_vec)
            # ★ 2026-09-17: 批次不一致时 `torch.cat` 只会抛
            #   "Sizes of tensors must match except in dimension 1"
            #   —— 完全看不出是哪个条件通路、各自多少，极难定位。
            #   这里显式检查并报出每个操作数的 batch，把模糊错误变成可诊断的。
            _bs = [int(p.shape[0]) for p in _parts]
            if len(set(_bs)) != 1:
                raise RuntimeError(
                    f"[factorized_cat] 条件操作数 batch 不一致: {_bs} "
                    f"(x batch={int(x.shape[0])}, g_tok="
                    f"{None if g_tok is None else int(g_tok.shape[0])}, "
                    f"y_callig={int(y_callig_in.shape[0])})。"
                    f" 通常是 CFG 路径里 x 被复制成 2B 但某个条件通路没跟上。")
            y_emb = self.cond_fusion(torch.cat(_parts, dim=-1))
        elif self.condition_fusion == "xl_highdim":
            # XL 高维条件：与 factorized_add 相同的 4-way 可控 mask（CFG 需要 uncond 维度）。
            # drop mask 已在 forward 顶部计算 (与骨架风格注入共享同一份)。
            if self.training and char_drop is not None:
                y_char = torch.where(char_drop, self.y_char_embedder.num_classes, y_char)
            e_callig = _e_callig()
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
        # 多层 REPA 捕获 (统一): return_intermediate_layers=(8,11) -> dict {8:feats, 11:feats}
        _repa_layers = None
        if return_intermediate_layers is not None:
            _repa_layers = (return_intermediate_layers
                            if isinstance(return_intermediate_layers, (list, tuple))
                            else (return_intermediate_layers,))
            intermediate_feats = {}
        # 兼容旧单层接口
        _repa_single = None
        if return_intermediate_layer is not None:
            _repa_single = int(return_intermediate_layer)
        # 预计算好的映射（__init__ 里建），不再每步重建
        _inj = {}
        if self.glyph_injections is not None and g_tok is not None:
            _inj = self._inj_map

        # ── 逐层注入的 context ───────────────────────────────────────────────
        # 默认 = 骨架 token(GlyphStyleCrossAttn 模式下位置在下面显式加)。
        # style_token_n>0 时拼接风格 token: 让**每一层**的 attention 都能直接
        # 看到书家风格, 而不是只通过"书家化骨架"间接进入(会被深层稀释)。
        # 每个 x 位置(query)因此可同时寻址: 局部字形(空间对应) + 书家风格。
        inject_ctx = g_tok
        if (getattr(self, "style_ctx_every_layer", False)
                and g_tok is not None and style_tokens is not None):
            # v15c: 骨架 token(+2D 位置) 与 K 个风格 token(+可学习 role) 拼成
            # 每层 xattn 的 K/V —— 风格在**每一层、每个空间位置**都可直接寻址。
            _Ng = g_tok.shape[1]
            inject_ctx = torch.cat(
                [g_tok + self.ctx_pos_g[:, :_Ng],
                 style_tokens + self.style_role.unsqueeze(0)], dim=1)
        elif getattr(self, "style_proj", None) is not None and g_tok is not None:
            _B = g_tok.shape[0]
            _D = g_tok.shape[-1]
            _Ng = g_tok.shape[1]
            _style = self.style_proj(e_callig).view(
                _B, self.n_style_token, _D) + self.style_role.unsqueeze(0)
            # 骨架 token 加 2D sincos 位置(空间对应), 风格 token 用可学习 role
            inject_ctx = torch.cat([g_tok + self.ctx_pos_g[:, :_Ng], _style], dim=1)

        if self.use_checkpoint:
            for i, block in enumerate(self.blocks):
                if _repa_layers is not None and i in _repa_layers:
                    x = block(x, c, rope=rope)
                    intermediate_feats[i] = x
                elif _repa_single is not None and i == _repa_single:
                    # Run this single block eagerly so its output can be captured for REPA.
                    x = block(x, c, rope=rope)
                    intermediate_feats = x
                else:
                    x = checkpoint(lambda *a: block(*a, rope=rope), x, c, use_reentrant=False)
                if i in _inj:
                    x = self.glyph_injections[_inj[i]](x, inject_ctx)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, c, rope=rope)
                if _repa_layers is not None and i in _repa_layers:
                    intermediate_feats[i] = x
                elif _repa_single is not None and i == _repa_single:
                    intermediate_feats = x
                if i in _inj:
                    # x = x*(1+s) + t，s/t 由 g_tok 经 zero-init Linear 产出。
                    # init 时恒等；梯度上 ∂out/∂g_tok = W = 0，因此这条路径
                    # 初期不给 glyph_embedder 梯度 —— 但输入层的
                    # x = x + glyph_scale * g_tok（glyph_scale=0.4 非零）
                    # 已提供直通梯度，故 glyph_embedder 从 step 0 即可学习。
                    x = self.glyph_injections[_inj[i]](x, inject_ctx)

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
        if _repa_layers is not None:
            return x, intermediate_feats   # dict {layer: feats} (多层 REPA)
        if return_intermediate_layer is not None:
            return x, intermediate_feats
        return x

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, g=None,
                         cfg_glyph=None, w_inter=0.0):
        """经典 2 路 CFG（默认），或委托给双轴 CFG。

        ★ 2026-09-17: 增加 `cfg_glyph` 参数。给了就走 `forward_with_2axis_cfg`
          （风格轴 × 内容轴各自独立），否则保持原有 2 路行为（**向后兼容**）。

        ⚠ 默认 2 路路径里 `g2 = cat([g, g])` —— **两半给同一个 g**，
          所以任何"只由 g 驱动"的通路（xattn / glyph_embedder）在
          `eps_cond − eps_uncond` 里**完全抵消，对 CFG 贡献恒为 0**。
          这是有意的（"CFG 只强化 callig 风格"），但会让组件结论失真：
          **xattn 这类通路在 cfg 引导下天然被绕过**（doc54 的 "xattn strict≈0" 即此）。
          要做内容轴引导必须走 `forward_with_2axis_cfg`，且训练时 `glyph_drop_prob > 0`。
        """
        if cfg_glyph is not None:
            return self.forward_with_2axis_cfg(
                x, t, y_callig, y_char,
                cfg_callig=cfg_scale, cfg_glyph=cfg_glyph, w_inter=w_inter, g=g)
        # Duplicate every sample: first copy conditional, second copy unconditional.
        original_bs = x.shape[0]
        # ★ 2026-09-17 诊断: 入口处各张量的 batch 必须一致，否则复制后仍不一致，
        #   最终在 factorized_cat 的 cat 里报一个看不出原因的 size 错。
        _in_bs = {"x": int(x.shape[0]), "t": int(t.shape[0]),
                  "y_callig": int(y_callig.shape[0]), "y_char": int(y_char.shape[0])}
        if g is not None:
            _in_bs["g"] = int(g.shape[0])
        if len(set(_in_bs.values())) != 1:
            raise RuntimeError(
                f"[forward_with_cfg] 入口 batch 不一致: {_in_bs}。"
                f" 复制成 2B 后仍会不一致 -> 后面 cat 会报模糊的 size 错。"
                f" 调用方（sample_latents 等）必须给同一段切片。")
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y_callig = torch.cat([y_callig, y_callig], dim=0)
        y_char = torch.cat([y_char, y_char], dim=0)
        uncond_callig = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
        y_callig_combined = torch.cat([y_callig[:original_bs], uncond_callig[original_bs:]], dim=0)
        if getattr(self, 'use_char_cond', True):
            uncond_char = torch.full_like(y_char, self.y_char_embedder.num_classes)
            y_char_combined = torch.cat([y_char[:original_bs], uncond_char[original_bs:]], dim=0)
        else:
            y_char_combined = y_char    # v10b: char 因子不存在, 值被 forward 忽略
        # 标准字形条件 g 始终全给(两半都用真实 g): 字形内容是正条件, CFG 只强化 callig 风格
        g2 = torch.cat([g, g], dim=0) if g is not None else None
        model_out = self.forward(x, t, y_callig_combined, y_char_combined, g=g2)
        if isinstance(model_out, tuple):
            model_out = model_out[0]  # skel_head 启用时 forward 返回 (主输出, skel_pred)，CFG 只取主输出
        # Apply CFG only on image latent channels (eps subspace), not canny/skel structure channels.
        eps, rest = model_out[:, :self.image_channels], model_out[:, self.image_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # ★ 2026-09-17: 原来写的是
        #     eps = torch.cat([half_eps, half_eps], dim=0)   # 白做一次 2B 分配
        #     out = torch.cat([eps, rest], dim=1)
        #     return out[:original_bs]                        # 又把后一半丢掉
        #   即先复制一份再截断 —— 纯浪费一次 2B×image_channels 的分配与拷贝。
        #   直接用 cond 那半的 rest(aux 通道取条件分支的值), 形状即 (B, C+rest)。
        #   语义完全一致 (返回行 = 前 original_bs 行)。
        out = torch.cat([half_eps, rest[:original_bs]], dim=1)
        return out

    def forward_with_2axis_cfg(self, x, t, y_callig, y_char,
                               cfg_callig=2.0, cfg_glyph=4.0, w_inter=0.0, g=None):
        """双轴 CFG: 风格轴(书家) × 内容轴(骨架 g) 各自独立强度。

        ★ 2026-09-17 修正。**旧实现是错的**（且零调用者，是死代码）:
          它把 "glyph 轴" 定义在 **`y_char`**（字符 ID）上, 并且
          `g4 = cat([g, g, g, g])` —— **g 在 4 个 pass 里完全相同**。
          后果: ① 对 `no_char_cond=True` 的架构(内容来自 g), glyph 轴是 **no-op**;
                ② g 在 `eps_cond − eps_uncond` 里**仍然完全抵消**,
                   所以 xattn / glyph_embedder 这类"只由 g 驱动"的通路对 CFG 贡献恒为 0。
          现在 glyph 轴改为**变化 g 本身**（置零 = "无骨架"分支）。

        4 个 pass 沿 batch 维拼成一次 forward:
          1. full    : (y_callig, g)     -> eps_full     两个条件都有
          2. style   : (y_callig, g=0)   -> eps_style    只有书家(风格), 无内容
          3. content : (null,     g)     -> eps_content  只有骨架(内容), 无风格
          4. uncond  : (null,     g=0)   -> eps_uncond   都没有

        Möbius / Product-of-Experts 组合:
          eps = eps_uncond
              + cfg_glyph  * (eps_content - eps_uncond)                 内容轴
              + cfg_callig * (eps_style   - eps_uncond)                 风格轴
              + w_inter    * (eps_full - eps_content - eps_style + eps_uncond)  交互项

        ⚠ **前提**: `g=0` 必须是模型训练时见过的条件 —— 即 `glyph_drop_prob > 0`。
          该 drop 的实现是 `g = g * keep`（整张样本置零），注释里明确写着
          "与 CFG 兼容: 丢弃时该样本等价于 uncond-g 分支"。
          若 `glyph_drop_prob == 0`，content 轴同样**未训练**，本函数会给出
          误导性结果（与 callig null 行是同一类问题）。
        """
        B = x.shape[0]
        x4 = torch.cat([x, x, x, x], dim=0)
        t4 = torch.cat([t, t, t, t], dim=0)

        null_c = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
        yc4 = torch.cat([y_callig, y_callig, null_c, null_c], dim=0)
        # char 通路保持不变（我们 no_char_cond=True，该值被 forward 忽略）
        yg4 = torch.cat([y_char, y_char, y_char, y_char], dim=0)

        if g is not None:
            zero_g = torch.zeros_like(g)
            g4 = torch.cat([g, zero_g, g, zero_g], dim=0)
        else:
            g4 = None

        model_out = self.forward(x4, t4, yc4, yg4, g=g4)
        if isinstance(model_out, tuple):
            model_out = model_out[0]

        eps4, rest4 = model_out[:, :self.image_channels], model_out[:, self.image_channels:]
        eps_full, eps_style, eps_content, eps_uncond = torch.split(eps4, B, dim=0)

        eps_guided = (
            eps_uncond
            + cfg_glyph * (eps_content - eps_uncond)
            + cfg_callig * (eps_style - eps_uncond)
            + w_inter * (eps_full - eps_content - eps_style + eps_uncond)
        )

        rest = rest4[:B]
        out = torch.cat([eps_guided, rest], dim=1)
        return out



def DiT_2Cond_M_2(**kwargs):
    # M (between S and Sp): h=432, d=12, heads=6 (~42M). 2026-09-12 用户裁定:
    # S/2 (384, 30M) seen 天花板 ~0.52 偏低; Sp/2 (512, 59M) 可达 0.67-0.76。
    # 取中间容量点验证 "容量-质量" 曲线。
    return DiT_2Cond(depth=12, hidden_size=432, patch_size=2, num_heads=6, **kwargs)


def DiT_2Cond_S320_2(**kwargs):
    # [v12+] 缩**宽度**探针: h=320, d=12, heads=5 (head_dim 64, 与 S/2 同)。
    # 动机见 docs/system/62_param_budget_derivation.md:
    #   - 渲染任务本质是"浅"的, 但 h=384 对 4 通道 latent 是 96x 扩张
    #   - FLOPs ~ d*h^2 = 12*320^2 = 0.549x M/2 (S/2 是 0.790x)
    #   - 保持 head_dim=64 与 S/2 一致, 使"宽度"成为唯一变量
    return DiT_2Cond(depth=12, hidden_size=320, patch_size=2, num_heads=5, **kwargs)


def DiT_2Cond_XS_2(**kwargs):
    # 更小变体：depth=8, hidden=384, 6 头, patch=2。参数约 20M（-35% vs S/2 的 30M），
    # 适合小数据量（3top30 仅 3.8 万图）防过拟合；transformer 层从 12→8。
    return DiT_2Cond(depth=8, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_2Cond_XS6_2(**kwargs):
    # [v12+] 更深一档的缩深度探针: d=6, h=384, 6 头。FLOPs 0.401x M/2。
    # 与 XS/2 组成 depth {12,8,6} 的阶梯, 用于定位"深度下界"。
    return DiT_2Cond(depth=6, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_2Cond_WS_2(**kwargs):
    # 宽体变体：类别多时加宽 hidden 而非加深 depth。depth=8, hidden=768, 12 头。
    # 参数约 70M（2.3× S/2），patch=2 保持精细位置编码，适合类别区分任务。
    return DiT_2Cond(depth=8, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_2Cond_S_2(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_2Cond_M_2(**kwargs):
    # M (between S and Sp): h=432, d=12, heads=6 (~42M). 2026-09-12 用户裁定:
    # S/2 (384, 30M) seen 天花板 ~0.52 偏低; Sp/2 (512, 59M) 可达 0.67-0.76。
    # 取中间容量点验证 "容量-质量" 曲线。
    return DiT_2Cond(depth=12, hidden_size=432, patch_size=2, num_heads=6, **kwargs)


def DiT_2Cond_Sp_2(**kwargs):
    # Sp (S-plus): h=512, d=12, heads=8, ~59M (1.8x S/2)。
    # 2026-09-08 欠拟合诊断 (train/strict velocity loss 全 t 桶高且平, 低 t 桶
    # 连训练原图都重建不动) 指向逐 token 表达容量不足 -> 加宽不加深:
    # g 残差流存活系数 1.72 已排除深度稀释; GT-g 同深度模型能解抄写任务
    # (follow-IoU3 0.57), 缺的是骨架->墨映射的表征带宽 (MLP/g_tok/adaLN 全 ×h)。
    return DiT_2Cond(depth=12, hidden_size=512, patch_size=2, num_heads=8, **kwargs)

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
    'DiT-2Cond-XS6/2': DiT_2Cond_XS6_2,
    'DiT-2Cond-WS/2': DiT_2Cond_WS_2,
    'DiT-2Cond-S/2': DiT_2Cond_S_2,
    'DiT-2Cond-S320/2': DiT_2Cond_S320_2,
    'DiT-2Cond-M/2': DiT_2Cond_M_2,
    'DiT-2Cond-Sp/2': DiT_2Cond_Sp_2,
    'DiT-2Cond-S/4': DiT_2Cond_S_4,
    'DiT-2Cond-S/8': DiT_2Cond_S_8,
    'DiT-2Cond-B/2': DiT_2Cond_B_2,
    'DiT-2Cond-B/4': DiT_2Cond_B_4,
}


