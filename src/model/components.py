# -*- coding: utf-8 -*-
"""components.py — DiT 的可复用组件（2026-10-13 从 dit.py 迁出，纯位移 ✓）

包含: modulate / TimestepEmbedder / LabelEmbedder / MultiStyleEmbedder /
      sincos 位置编码 3 件套 / ZeroCrossAttention / GlyphStyleCrossAttn /
      CalligStyleCrossAttn / StyleHierarchy / ScriptGlyphFiLM / SpatialStyleFiLM /
      LowRankSpatialStyleFiLM / LocalStyleGlyphAdapter

dit.py 会**原样再导出**这些名字，因此 `from .dit import LabelEmbedder` 等既有引用不受影响 ✓
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

log = logging.getLogger(__name__)


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
            # [infra 2026-10-04] 去掉 if null_mask.any() 数据依赖分支: .any() 在
            # compiled 区域 graph break, 批间翻转让 dynamo 反复重编译, ~250 步打穿
            # accumulated_cache_size_limit 256 -> 静默回退 eager -> eager ~22G -> OOM。
            # 无分支 where 语义等价 (全 False = 恒等选择), 开销一次 (B,D) select。
            null_mask = (labels == self.num_classes)
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

    def __init__(self, d_model, num_heads, grid_size=16, q_pos=False, attn_tau=1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # ★ 2026-10-03: 注意力温度 τ。softmax(qkᵀ/√d/τ) ≡ softmax((q/τ)kᵀ/√d),
        #   所以只需把 q 除以 τ, **零额外开销**。τ<1 = 锐化 / hard routing。
        #   动机(探针实测 tools/probe_style_routing.py): 风格 token 的独占熵 τ=1.0 时 0.974(均匀塔缩),
        #   τ=0.1 时降到 0.397(真路由)。根因不是 token 共线, 而是初始化时 q/k 投影随机 -> logits 无差异。
        #   τ=1.0 = 旧行为逐位等价。
        self.attn_tau = float(attn_tau)
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
        if self.attn_tau != 1.0:
            q = q / self.attn_tau      # = softmax(qkᵀ/√d/τ)
        # ★ [2026-10-06] 层级敏感度探针（默认关；只有 --probe-inject-every > 0 时
        #   InjectLayerProbe 才会把本模块的 .probe 置 True）。
        #   显式算 attention 并记录**空间寻址**统计 —— 比"注意力锐度"对症:
        #   diag = Q 的第 k 个 token 给 K 中**同一网格位置**骨架 token 的权重
        #   (随机均匀 = 1/N = 0.0039; 越大学得越像"看着同一位置写")。
        #   正常训练绝不走这里（显式 softmax 更慢更吃显存）。
        if getattr(self, "probe", False):
            _att = ((q @ k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)).softmax(dim=-1)
            with torch.no_grad():
                _a = _att.detach()
                self.probe_stats = {
                    "diag": float(_a.diagonal(dim1=-2, dim2=-1).mean()),
                    "entropy": float(-(_a.clamp_min(1e-9).log() * _a).sum(-1).mean()),
                    "maxw": float(_a.max(dim=-1)[0].mean()),
                    "n_key": int(_a.shape[-1]),
                }
            out = _att @ v
        else:
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

    def __init__(self, d_model, num_heads, attn_tau=1.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_tau = float(attn_tau)   # τ<1 = 锐化, 见 ZeroCrossAttention
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
        if self.attn_tau != 1.0:
            q = q / self.attn_tau
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

    def __init__(self, hidden_size, num_heads, grid_size=16, attn_tau=1.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_tau = float(attn_tau)   # τ<1 = 锐化, 见 ZeroCrossAttention
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
        if self.attn_tau != 1.0:
            q = q / self.attn_tau
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, D)
        return g_tok + self.out_proj(out)


#################################################################################
#      S2 (2026-09-22): 三层语义分解 + 局部风格-骨架引导                          #
#################################################################################
#
# 设计文档: docs/922/20_style_encoding.md (表怎么分级) + docs/922/30_injection.md
#           (adaLN vs cross-attn) + docs/922/50_implementation.md (落地说明)
#
# 一句话: 全局笔法继续走 adaLN；局部书写引导由**新增的轻量通路**承担。
#         这里**不替换** adaLN，也不再用"假风格 token"当 K/V。
#################################################################################


class StyleHierarchy(nn.Module):
    """S2 的层级部分：pair 交互残差 + 书体层（书家主效应复用既有表）。

    ## 为什么书家主效应不新建一张表

    既有的 ``y_callig_embedder``（LabelEmbedder / MultiStyleEmbedder）已经承载了
    四套机制，全部复用可以零风险：

      * SupCon 预训练加载（``--callig-emb-pretrained``）
      * CFG null 行语义（``label == num_classes``）
      * 冻结（``freeze_table()``）
      * few-shot 新增行（``--train-only-new-callig`` / ``--init-new-callig``）
        ← **这一条最关键**：S2 的卖点之一就是"新书家只学 128 维主效应行"，
          复用后该机制直接可用，不用重写。

    所以本模块只负责**新增的两层**：

    .. code-block:: text

        e_style  = e_callig(主效应) + α · E_pair[pair]     α init 0.1，可学习
        e_script = E_script[script]

    ## 残差为什么能"自动回退"

    ``E_pair`` **zero-init** 且 ``α`` 很小：训练初期 pair 项恒为 0，
    模型等价于"纯书家主效应"（= 已验证的 v13 配方的超集），可安全 resume。
    稀疏 pair（数据里 15 个 <50 样本，min=1）梯度微弱 → 残差学不动 →
    **自动退化为主效应兜底**，不需要任何硬编码回退逻辑。

    ## drop 语义

    callig 被 4-way dropout 丢弃时，pair 项**整项置零**（不是换成 null 行），
    于是 ``e_style = null_callig + 0`` —— uncond 分支保持纯净。
    """

    def __init__(self, num_pairs, num_scripts, style_dim, script_dim=64,
                 pair_init="zero", pair_residual=1):
        super().__init__()
        self.num_pairs = int(num_pairs)
        self.num_scripts = int(num_scripts)
        self.style_dim = int(style_dim)
        self.script_dim = int(script_dim)
        self._pair_init = str(pair_init)
        self.E_pair = nn.Embedding(self.num_pairs, self.style_dim)
        self.E_script = nn.Embedding(self.num_scripts, self.script_dim)
        # null(书体) 行：script 缺失/被 drop 时用（当前 4-way drop 不丢 script，
        # 但为 CFG 与未来扩展保留）。
        self.null_script = nn.Parameter(torch.zeros(self.script_dim))
        # α 是残差门控。0.1 起步：让主效应先站稳，再逐步放开交互项。
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # pair_residual=0 -> 冻结 α 且置 0：残差恒为 0，等价于"只有书体层"。
        # 用来做**参数个数完全相同**的消融（S2-a），避免"加没加这一层"说不清。
        # ⚠ 注意 ∂L/∂α = ∂L/∂e_style · E_pair 在 E_pair zero-init 时为 0 ->
        #   α 的梯度有**一步延迟**（第一步只更新 E_pair，第二步起 α 才动）。
        #   这不是死锁，但设计上要知道。
        self.pair_residual = int(pair_residual)
        if self.pair_residual == 0:
            with torch.no_grad():
                self.alpha.zero_()
            self.alpha.requires_grad_(False)
        nn.init.normal_(self.E_script.weight, std=0.02)
        if str(pair_init) == "zero":
            nn.init.zeros_(self.E_pair.weight)
        else:
            nn.init.normal_(self.E_pair.weight, std=0.02)

    def forward(self, e_callig, pair_id, script_id, callig_drop=None):
        """返回 ``(e_style, e_script)``。

        ``pair_id`` / ``script_id`` 可为 None（数据集没给）→ 对应项按"缺失"处理。
        """
        B = e_callig.shape[0]
        dev, dt = e_callig.device, e_callig.dtype

        if pair_id is None:
            e_pair = torch.zeros_like(e_callig)
        else:
            pair_id = pair_id.to(dev)
            _bad = pair_id >= self.num_pairs
            if callig_drop is not None:
                _bad = _bad | callig_drop.to(dev).to(torch.bool)
            _safe = pair_id.clamp(0, self.num_pairs - 1)
            e_pair = self.E_pair(_safe).to(dt)
            e_pair = torch.where(_bad.unsqueeze(-1), torch.zeros_like(e_pair), e_pair)

        e_style = e_callig + self.alpha.to(dt) * e_pair

        if script_id is None:
            e_script = self.null_script.to(dt).unsqueeze(0).expand(B, -1)
        else:
            script_id = script_id.to(dev)
            _bad_s = script_id >= self.num_scripts
            _safe_s = script_id.clamp(0, self.num_scripts - 1)
            e_script = self.E_script(_safe_s).to(dt)
            e_script = torch.where(
                _bad_s.unsqueeze(-1),
                self.null_script.to(dt).unsqueeze(0).expand_as(e_script),
                e_script)
        return e_style, e_script


class ScriptGlyphFiLM(nn.Module):
    """通路 B1：书体 FiLM on ``g_tok``（结构轴）。

    书体（楷/行/隶）是**结构**语义，与 ``g`` 同域 —— ``g`` 本身就是按书体渲染的
    标准骨架，两者冗余度高。所以书体**不进 adaLN**（那会变成"同一个信号喂两遍"，
    12ch 失败的教训），而是去**调制骨架**。

    ``zero-init`` → step 0 时 ``γ=β=0`` → 恒等映射，可从任何已训 ckpt 安全续跑。

    ⚠ **加性 β 必须乘 keep**：``g_tok`` 在 glyph_drop 时会先被置零表示"无骨架"，
    若此时再加上非零 β，被丢弃的样本会重新获得条件信号 → **uncond 分支被污染**。
    乘性 γ 不受影响（``0*(1+γ)=0``）。
    """

    def __init__(self, script_dim, d_model):
        super().__init__()
        self.film = nn.Linear(int(script_dim), 2 * int(d_model))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, g_tok, e_script, keep=None):
        gamma, beta = self.film(e_script).chunk(2, dim=-1)
        gamma = gamma.to(g_tok.dtype).unsqueeze(1)
        beta = beta.to(g_tok.dtype).unsqueeze(1)
        if keep is not None:
            beta = beta * keep.view(-1, 1, 1).to(g_tok.dtype)
        return g_tok * (1 + gamma) + beta


class SpatialStyleFiLM(nn.Module):
    """通路 B2（Phase 1，比 cross-attn 便宜）：**逐位置**的局部 FiLM。

    .. code-block:: text

        γ_i, β_i = f( e_cond, g_tok_i, pos_i )
        g_tok_i' = g_tok_i · (1 + γ_i) + β_i · keep

    与 :class:`ScriptGlyphFiLM` 的区别：那里的 γ/β 对**所有 token 相同**（全局），
    这里**每个 token 不同**（局部）—— 已经能表达
    "同一风格下不同局部位置有不同处理" / "不同风格下同一局部位置有不同处理"。

    成本远低于 cross-attn：256 个 token × ~106K 参数（D=384, rank=64）
    ≈ 27M MACs/样本，约为全模型 FLOPs 的 1%。

    为什么先做它再做 cross-attn：如果局部性有效但 FiLM 就够了，就不必付
    attention 的钱；只有 FiLM 明显不够时才升级到 :class:`LocalStyleGlyphAdapter`。
    """

    def __init__(self, cond_dim, d_model, rank=64):
        super().__init__()
        self.d_model = int(d_model)
        # 输入 = [e_cond; g_tok_i; pos_i]
        self.net = nn.Sequential(
            nn.Linear(int(cond_dim) + 2 * int(d_model), int(rank)),
            nn.SiLU(),
            nn.Linear(int(rank), 2 * int(d_model)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, g_tok, e_cond, pos, keep=None):
        B, N, D = g_tok.shape
        if int(e_cond.shape[0]) != B:
            raise RuntimeError(
                f"[SpatialStyleFiLM] batch 不一致: g_tok={B} vs e_cond="
                f"{int(e_cond.shape[0])}（通常是 CFG 复制 2B 时条件没跟上）")
        _cond = e_cond.to(g_tok.dtype).unsqueeze(1).expand(B, N, -1)
        _pos = pos.to(g_tok.dtype)
        if _pos.shape[1] != N:
            raise ValueError(
                f"[SpatialStyleFiLM] pos token 数 {_pos.shape[1]} != g_tok {N}")
        z = torch.cat([_cond, g_tok, _pos.expand(B, N, -1)], dim=-1)
        gamma, beta = self.net(z).chunk(2, dim=-1)
        if keep is not None:
            beta = beta * keep.view(-1, 1, 1).to(g_tok.dtype)
        return g_tok * (1 + gamma) + beta


class LowRankSpatialStyleFiLM(nn.Module):
    """低秩逐位置风格 × 几何乘法, 作用在 ``g_tok`` 上。

    .. code-block:: text

        s        = P · e_style              (r,)    只看书家
        q_i      = Q · g_tok_i              (r,)    只看这个格子的几何
        u_i      = s ⊙ q_i                          乘积项, 两边缺一不可
        γ_i, β_i = MLP(u_i, pos_i)
        g_tok_i' = g_tok_i · (1 + γ_i) + β_i · keep

    与 :class:`SpatialStyleFiLM` 的差别: 那里把 ``(e_style, g_tok_i, pos)``
    拼接后交给线性层, 线性层**可以**学出乘法, 但没有被逼着做, 容量会花在
    旁路上。这里乘法写死, 同一个书家在不同格子上 ``q_i`` 不同, ``u_i`` 就不同。

    不加辅助损失, 不加 GT 实例骨架。主干始终吃 ``g_tok'``, 全局 adaLN 不动。

    输出层采用小随机初始化，方便梯度传到 P/Q。单独一个零输出层本身仍可
    收到梯度，只会在首步阻断其上游；不能将它误判为永久死锁。连续的零门控
    与零残差才可能互相阻断。实际梯度还取决于结构编码器和下游损失。

    ``β`` 必须乘 ``keep``。glyph_drop 时 ``g_tok`` 是 0, 乘性 γ 自然消失,
    不加这句的话 β 会把丢弃样本重新灌成有条件, CFG 的 unconditional 分支被污染。
    """

    def __init__(self, style_dim, d_model, rank=32):
        super().__init__()
        self.rank = int(rank)
        self.d_model = int(d_model)
        self.P = nn.Linear(int(style_dim), self.rank, bias=False)
        self.Q = nn.Linear(int(d_model), self.rank, bias=False)
        self.film_net = nn.Sequential(
            nn.Linear(self.rank + int(d_model), self.rank),
            nn.SiLU(),
            nn.Linear(self.rank, 2 * int(d_model)),
        )
        # 把风格化后的 token 解码回 VAE latent。一个 token 管 2×2 个 latent 像素,
        # 所以输出 4×4=16 维, 再排成 (4,32,32)。只在单独训头时用, 主干前向不调用。
        self.to_latent = nn.Linear(int(d_model), 4 * 2 * 2)
        self.reset_output()

    def reset_output(self):
        """最后一层小随机, 其余保持构造时的默认初始化。

        ``DiT_2Cond.initialize_weights`` 里的 ``_basic_init`` 会把所有 Linear
        重新 xavier 一遍, 所以这里必须能被再调一次。
        """
        nn.init.normal_(self.film_net[-1].weight, std=0.002)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, g_tok, e_style, pos, keep=None, *, cache_styled=True):
        B, N, D = g_tok.shape
        if int(e_style.shape[0]) != B:
            raise RuntimeError(
                f"[LowRankSpatialStyleFiLM] batch 不一致: g_tok={B} vs e_style="
                f"{int(e_style.shape[0])} (通常是 CFG 把 x 复制成 2B 但条件没跟上)")
        if int(e_style.shape[-1]) != self.P.in_features:
            raise RuntimeError(
                f"[LowRankSpatialStyleFiLM] e_style 维度 {int(e_style.shape[-1])} "
                f"!= 构造时的 {self.P.in_features}")
        if pos.shape[1] != N:
            raise ValueError(
                f"[LowRankSpatialStyleFiLM] pos token 数 {pos.shape[1]} != g_tok {N}")
        s = self.P(e_style.to(g_tok.dtype))                       # (B, r)
        q = self.Q(g_tok)                                         # (B, N, r)
        u = s.unsqueeze(1) * q                                    # (B, N, r)
        z = torch.cat([u, pos.to(g_tok.dtype).expand(B, N, -1)], dim=-1)
        gamma, beta = self.film_net(z).chunk(2, dim=-1)
        if keep is not None:
            beta = beta * keep.to(g_tok.dtype).view(-1, 1, 1)
        styled = g_tok * (1 + gamma) + beta
        if cache_styled:
            self._last_styled = styled
        out = styled
        if keep is not None:
            # 乘性 γ 在 g_tok=0 时自然消失, 但 β 是加性的。keep=0 的样本
            # 必须整行归零, 否则丢弃的骨架条件被重新灌回来, CFG 的
            # unconditional 分支不再是无条件。
            out = out * keep.to(g_tok.dtype).view(-1, 1, 1)
        return out

    def decode_latent(self):
        """forward 之后调用, 把风格化 token 还原成 (B,4,32,32) latent。"""
        h = self._last_styled
        B, N, _ = h.shape
        grid = int(round(N ** 0.5))
        pix = self.to_latent(h).view(B, grid, grid, 2, 2, 4)
        return pix.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, 4, grid * 2, grid * 2)


class LocalStyleGlyphAdapter(nn.Module):
    """通路 B3（Phase 2）：**局部风格-骨架** cross-attention adapter。

    ## 与两个历史方案的本质区别

    旧 ``CalligStyleCrossAttn``（v15b）::

        Q = 骨架 token,  K/V = K 个**风格 token**

    → K 个风格 token 间余弦 **0.884**（几乎共线），"多模态"是假的，
      attention 没有可寻址的内容，最后退化成昂贵的全局调制（实测只值 +0.0016）。

    旧 ``style_ctx_every_layer``（v15c）::

        K token 拼进**每层** xattn context，成本 +33%，独立口径**最差**。

    本模块::

        Q = x 或 g_tok（**带位置**）
        K/V = **局部骨架 token**（**带位置**）        ← 空间证据是真的
        style 通过 FiLM 调制 Q/K                      ← 风格决定"看哪里"

    → 风格**不作为被查询对象**，而是**改变注意力分布**。
      不再依赖 K 个假风格 token。

    ## 语义

    | 书体/书家 | 关注的局部位置 |
    |---|---|
    | 隶书 | 横向笔画末端、波磔位置 |
    | 楷书 | 起收笔的方整、结构均衡 |
    | 行书 | 转折、牵丝、粗细变化 |

    ## 三个必须遵守的工程约束

    1. **Q 必须有位置编码**（旧 ZeroCrossAttention 只给 K/V 加 → 空间寻址退化成
       内容寻址）。这里 Q/K 都加同一份 2D sincos。
    2. **out_proj 与 style_qk 末层 zero-init** → step 0 恒等，可安全 resume。
    3. **输出必须乘 keep**（glyph_drop 的样本骨架是零，但 attention 会从
       这些 token 里聚合出非零值 → 污染 uncond 分支）。

    ⚠ 实现修正：风格 γ/β 的形状是 ``(B, D)``，而 q/k 是 ``(B, H, N, hd)``。
    直接 ``view(B,1,1,D)`` 广播会在 ``D vs hd`` 上失配（早前的参考实现有此 bug）。
    必须 reshape 成 ``(B, H, 1, hd)``。
    """

    def __init__(self, d_model, cond_dim, num_heads=4, rank=64, window=0,
                 grid_size=16):
        super().__init__()
        d_model = int(d_model)
        assert d_model % int(num_heads) == 0, "d_model 必须被 num_heads 整除"
        self.num_heads = int(num_heads)
        self.head_dim = d_model // self.num_heads
        self.d_model = d_model
        grid_size = int(round(grid_size))
        self.grid_size = grid_size

        self.cond_norm = nn.LayerNorm(int(cond_dim))
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 风格 → Q/K 的 γ/β（4 组：γ_q, β_q, γ_k, β_k）
        self.style_qk = nn.Sequential(
            nn.Linear(int(cond_dim), int(rank)),
            nn.SiLU(),
            nn.Linear(int(rank), 4 * d_model),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.style_qk[-1].weight)
        nn.init.zeros_(self.style_qk[-1].bias)
        # sigmoid(0)=0.5, 乘 cap 0.2 = 初值 0.1。硬顶 0.2,
        # 防止 out_proj 一离开 0 就把残差流改掉六成。
        self.res_cap = 0.2
        self.res_logit = nn.Parameter(torch.zeros(()))

        # 窗口注意力（可选）：每个位置只看局部邻域，更像"局部书写指引"，
        # 也更不容易退化成全局平均。window=0 → 全局 256-token attention。
        self.window = int(window)
        if self.window > 0:
            N = grid_size * grid_size
            idx = torch.arange(N)
            r, c = idx // grid_size, idx % grid_size
            half = self.window // 2
            m = ((r.unsqueeze(1) - r.unsqueeze(0)).abs() <= half) & \
                ((c.unsqueeze(1) - c.unsqueeze(0)).abs() <= half)
            # 对角线恒为 True，保证没有全 False 的行（否则 softmax 出 NaN）
            m = m | torch.eye(N, dtype=torch.bool)
            # ⚠ 与 ctx_pos_g 同一个坑：register_buffer 对**已存在的属性名**会抛
            #   KeyError: "attribute already exists"。所以不能先写 self.win_mask = None。
            if "win_mask" in self.__dict__:
                del self.__dict__["win_mask"]
            self.register_buffer("win_mask", m.unsqueeze(0).unsqueeze(0),
                                 persistent=False)
        else:
            self.win_mask = None

    def forward(self, q_src, g_tok, e_cond, pos, keep=None):
        """q_src: (B,N,D) 查询源（x 或 g_tok）; g_tok: (B,N,D) 骨架证据。"""
        B, N, D = q_src.shape
        Nc = g_tok.shape[1]
        H, hd = self.num_heads, self.head_dim
        # ★ 显式 batch 校验：本项目历史上多次因"某个条件通路没跟上 2B"而在
        #   cat/view 处报一个看不出原因的形状错（见 factorized_cat 处的同类注释）。
        _bs = {"q_src": B, "g_tok": int(g_tok.shape[0]), "e_cond": int(e_cond.shape[0])}
        if len(set(_bs.values())) != 1:
            raise RuntimeError(
                f"[LocalStyleGlyphAdapter] batch 不一致: {_bs}。"
                f" 通常是 CFG 路径里 x 被复制成 2B 但条件没跟上。")

        q_in = self.norm_q(q_src + pos.to(q_src.dtype))
        kv_in = self.norm_kv(g_tok + pos.to(g_tok.dtype))

        q = self.q_proj(q_in).view(B, N, H, hd).transpose(1, 2)          # (B,H,N,hd)
        k = self.k_proj(kv_in).view(B, Nc, H, hd).transpose(1, 2)        # (B,H,Nc,hd)
        v = self.v_proj(kv_in).view(B, Nc, H, hd).transpose(1, 2)

        gq, bq, gk, bk = self.style_qk(self.cond_norm(e_cond)).chunk(4, dim=-1)
        # ⚠ (B,D) -> (B,H,1,hd) 才能与 (B,H,N,hd) 正确广播
        gq = gq.view(B, H, hd).unsqueeze(2).to(q.dtype)
        bq = bq.view(B, H, hd).unsqueeze(2).to(q.dtype)
        gk = gk.view(B, H, hd).unsqueeze(2).to(k.dtype)
        bk = bk.view(B, H, hd).unsqueeze(2).to(k.dtype)
        # 风格缩放夹在 [-1, 1]。不夹的话, 窗口里只有约 25 个 token,
        # logits 在训练早期就会把 softmax 打溢出。
        q = q * (1 + gq.clamp(-1, 1)) + bq.clamp(-1, 1)
        k = k * (1 + gk.clamp(-1, 1)) + bk.clamp(-1, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)     # (B,H,N,Nc)
        if self.win_mask is not None and Nc == self.win_mask.shape[-1]:
            scores = scores.masked_fill(~self.win_mask.to(torch.bool),
                                        torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        # 固定 0.1, 不再用可学习门。可学习门在 800 步时仍把残差改掉一半。
        if keep is not None:
            out = out * keep.view(-1, 1, 1).to(out.dtype)
        return q_src + 0.1 * out


