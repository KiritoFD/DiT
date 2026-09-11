# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
#####
#  这是FIM_VIM 用Vision Mamba替换了Attention
# 在原先的基础上添加了rope result_4_ROPE
# 同时，计算了残差
# 并且添加了之前x的norm/scale,shift
# 在calligraphy的基础上，添加3个label
#####
from functools import partial
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import torch.nn as nn
import numpy as np
import math
from mamba_ssm import Mamba2
from timm.models.layers import DropPath, to_2tuple
from utils.rope import VisionRotaryEmbeddingFast
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from utils.fasterkan import FasterKAN
from utils.stroke import get_ch_strokes_tensor, max_ch_strokes, max_stroke_pts
from einops import rearrange
import time

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


device = "cuda:0"


######
# VMamba#
#####
class VisionMambaBlock(nn.Module):
    """
    这个是VisionMamba的Block
    """

    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # if self.fused_add_norm:
        #     assert RMSNorm is not None, "RMSNorm import fails"
        #     assert isinstance(
        #         self.norm, (nn.LayerNorm, RMSNorm)
        #     ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        # else:
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
        #     if residual is None:
        #         hidden_states, residual = fused_add_norm_fn(
        #             hidden_states,
        #             self.norm.weight,
        #             self.norm.bias,
        #             residual=residual,
        #             prenorm=True,
        #             residual_in_fp32=self.residual_in_fp32,
        #             eps=self.norm.eps,
        #         )
        #     else:
        #         hidden_states, residual = fused_add_norm_fn(
        #             self.drop_path(hidden_states),
        #             self.norm.weight,
        #             self.norm.bias,
        #             residual=residual,
        #             prenorm=True,
        #             residual_in_fp32=self.residual_in_fp32,
        #             eps=self.norm.eps,
        #         )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_vision_mamba_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"dtype": torch.float, "device": device}
    # Mamba2 需要保证 d_model * expand / headdim = multiple of 8  
    # https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940
    mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        # nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = VisionMambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.device = device
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
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
        )
        # 额外指定了 device
        if self.device:
            freqs = freqs.to(self.device)
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
    三个张量会进行拼接并映射到初始维度
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table1 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.embedding_table2 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.embedding_table3 = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

        self.linear = nn.Linear(hidden_size * 3, hidden_size)
        self.num_classes = num_classes
        # print(num_classes)
        self.dropout_prob = dropout_prob

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
        new_labels = labels.T
        # print(labels.shape)
        if (train and use_dropout) or (force_drop_ids is not None):
            new_labels = [self.token_drop(label, force_drop_ids) for label in new_labels]

        assert torch.max(new_labels[0]) < self.num_classes + 1, f"Index out of range: {torch.max(new_labels[0])}"
        assert torch.max(new_labels[1]) < self.num_classes + 1, f"Index out of range: {torch.max(new_labels[1])}"
        assert torch.max(new_labels[2]) < self.num_classes + 1, f"Index out of range: {torch.max(new_labels[2])}"
        calligraphy_embedding = self.embedding_table1(new_labels[0])
        font_embedding = self.embedding_table2(new_labels[1])
        char_embedding = self.embedding_table3(new_labels[2])
        # print(calligraphy_embedding.shape, font_embedding.shape, char_embedding.shape)
        cat_label = torch.cat((calligraphy_embedding, font_embedding, char_embedding), dim=1)
        res = self.linear(cat_label)
        # print(res.shape)
        return res


class FeatureEmbedder(nn.Module):
    def __init__(self, in_channel=2048, out_channel=1024, in_size=16, dropout_prob=0):
        super().__init__()
        kernel_size = int(math.sqrt(in_size))
        stride = kernel_size
        assert (kernel_size * kernel_size == in_size)

        # 创建一个通用的卷积层构建函数
        def create_conv_layers(in_ch, out_ch, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=(in_ch + out_ch) // 2, kernel_size=kernel_size,
                          stride=stride),
                nn.Conv2d(in_channels=(in_ch + out_ch) // 2, out_channels=out_ch, kernel_size=kernel_size,
                          stride=stride)
            )

        # 使用该函数来生成所需的卷积层
        # self.calligrapher_conv = create_conv_layers(in_channel, out_channel, kernel_size)
        # self.font_conv = create_conv_layers(in_channel, out_channel, kernel_size)
        # self.char_conv = create_conv_layers(in_channel, out_channel, kernel_size)
        self.conv = create_conv_layers(in_channel, out_channel, kernel_size)

        # self.linear = nn.Linear(out_channel * 3, out_channel)
        self.use_drop_out = dropout_prob > 0  # TODO:drop_out

    def feature_drop(self, labels, train, force_drop_ids=None):
        pass

    def forward(self, labels, train, force_drop_ids=None):
        res = self.conv(labels)
        res = res.squeeze(-1).squeeze(-1)
        # print(res.shape)
        return res


class StrokeEmbedder(nn.Module):
    def __init__(self, max_ch_strokes, max_stroke_pts, out_dim, device):
        super().__init__()
        self.device = device
        self.hidden_size = out_dim

    def forward(self, x):
        """ch:中文字list"""
        padding_size = self.hidden_size - x.shape[-1]
        x_padded = F.pad(x, (0, padding_size))
        return x_padded


#################################################################################
#                                 Core Moyun Model                              #
#################################################################################


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MoyunBlock(nn.Module):
    """
    A Moyun block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,  # attention 废弃
                 layer_idx,
                 use_mamba=True,
                 use_kan=True,
                 use_stroke=True,
                 mlp_ratio=4.0,
                 **block_kwargs):
        super().__init__()
        self.use_mamba = use_mamba  # use mamba or Transformer
        self.use_kan = use_kan  # use kan or mlp
        self.use_stroke = use_stroke
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-3)
        if self.use_mamba:
            factory_kwargs = {"dtype": torch.float}
            self.attn_mamba = create_vision_mamba_block(hidden_size, layer_idx=layer_idx, **factory_kwargs)  # VIM
        else:
            self.attn_mamba = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                **block_kwargs)
        if self.use_stroke:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-3)
            self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-3)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if self.use_kan:
            self.mlp = FasterKAN([hidden_size, mlp_hidden_dim, hidden_size])
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, res, c, stroke):
        if self.use_mamba:
            # 如果该模块使用mamba
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            mamba_res = self.attn_mamba(modulate(self.norm1(x), shift_msa, scale_msa), res, None)
            x = x + gate_msa.unsqueeze(1) * mamba_res[0]
            res = mamba_res[1]
            if self.use_stroke:
                tmp = self.norm2(x)
                tmp = rearrange(tmp, "b p e -> p b e")
                stroke = rearrange(stroke, "b s e -> s b e")
                # print(stroke.shape, tmp.shape)
                tmp, _ = self.cross_attention(tmp, stroke, stroke)
                tmp = rearrange(tmp, "p b e -> b p e")
                x = x + tmp
            tmp = modulate(self.norm3(x), shift_mlp, scale_mlp)
            tmp = rearrange(tmp, "b p e -> (b p) e")
            tmp = self.mlp(tmp)
            tmp = rearrange(tmp, "(b p) e -> b p e", b=x.shape[0])
            x = x + gate_mlp.unsqueeze(1) * tmp
            return x, res
        else:
            # 如果该模块使用 Transformer
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            # print(shift_msa)
            # shift_msa *= 0.09
            shift_msa = shift_msa * 1.2
            scale_msa = scale_msa * 1.2
            gate_msa = gate_msa * 1.2
            shift_mlp = shift_mlp * 1.2
            scale_mlp = scale_mlp * 1.2
            gate_mlp = gate_mlp * 1.2
            # print(shift_msa)
            x = x + gate_msa.unsqueeze(1) * self.attn_mamba(modulate(self.norm1(x), shift_msa, scale_msa))
            if self.use_stroke:
                tmp = self.norm2(x)
                tmp = rearrange(tmp, "b p e -> p b e")
                stroke = rearrange(stroke, "b s e -> s b e")
                tmp, _ = self.cross_attention(tmp, stroke, stroke)
                tmp = rearrange(tmp, "p b e -> b p e")
                x = x + tmp
            tmp = modulate(self.norm3(x), shift_mlp, scale_mlp)
            tmp = rearrange(tmp, "b p e -> (b p) e")
            tmp = self.mlp(tmp)
            tmp = rearrange(tmp, "(b p) e -> b p e", b=x.shape[0])
            x = x + gate_mlp.unsqueeze(1) * tmp
            return x, None


class Moyun(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,  # TODO: 设置超参调整Transformer和Mamba
            num_heads=16,
            mlp_ratio=4.0,
            pt_hw_seq_len=14,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
            device=None,
            if_rope=False,
            if_rope_residual=False,
            use_mamba=False,
            use_kan=False,
            use_stroke=False,
            use_feature=True,
            repa_dep=12
    ):
        super().__init__()
        self.device = device
        self.use_stroke = use_stroke
        self.use_feature = use_feature
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True).to(device)
        num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(hidden_size, device=device)
        if use_stroke:
            self.stroke_embedder = StrokeEmbedder(max_ch_strokes=num_patches,
                                                  max_stroke_pts=max_stroke_pts,
                                                  out_dim=hidden_size,
                                                  device=self.device
                                                  )  # TODO:Stroke
        if use_feature:
            self.y_embedder = FeatureEmbedder(in_channel=hidden_size * 3, out_channel=hidden_size)
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        if if_rope:
            half_head_dim = hidden_size // 2
            hw_seq_len = input_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )
        else:
            self.rope = None

        self.repa_depth = repa_dep
        assert (repa_dep <= depth)
        self.blocks = nn.ModuleList([
            MoyunBlock(hidden_size,
                       num_heads,
                       layer_idx=i,
                       mlp_ratio=mlp_ratio,
                       use_mamba=use_mamba,
                       use_kan=use_kan,
                       use_stroke=use_stroke
                       )
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # print(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if not self.use_feature:
            nn.init.normal_(self.y_embedder.embedding_table1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.embedding_table2.weight, std=0.02)
            nn.init.normal_(self.y_embedder.embedding_table3.weight, std=0.02)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unpatch_feature(self, x):
        """
        x: (N, T, C)
        feature: (N, C, H, W)
        """
        c = self.hidden_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c))
        feature = torch.einsum('nhwc->nchw', x)
        return feature

    def forward(self, x, t, y, stroke):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        # print("test")
        x = self.x_embedder(x)
        if not self.if_rope:
            x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        if self.use_stroke:
            stroke = self.stroke_embedder(stroke)
        else:
            stroke = None
        # print(t.shape)
        # print(y.shape)
        c = t + y  # (N, D)
        hidden_states = x
        residual = None
        repa_res = None
        dep_cnt = 1

        

        for layer in self.blocks:
            # print(self.if_rope, self.if_rope_residual)
            if self.if_rope:
                hidden_states = self.rope(hidden_states)
            if self.if_rope_residual and (residual is not None):
                residual = self.rope(residual)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                c,
                stroke
            )
            # TODO:repa的返回目前是都对上了的，如果没对上后面加MLP
            if dep_cnt == self.repa_depth:
                repa_res = hidden_states
            dep_cnt += 1

        
        
        x = hidden_states
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # print(f"before{repa_res.shape=}")
        repa_res = self.unpatch_feature(repa_res)
        # print(f"after{repa_res.shape=}")
        # REPA新添加了返回值
        
        return x, repa_res

    def forward_with_cfg(self, x, t, y, stroke, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        model_out, _ = self.forward(combined, t, y, stroke)  # 这里因为新加feature，忽略feature返回
        
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

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

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
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
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   Moyun Configs                                  #
#################################################################################
def test_models(**kwargs):  # L_2
    return Moyun(depth=24,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 **kwargs)


def test_models_nofeature(**kwargs):  # L_2
    return Moyun(depth=24,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)


def test_models_nofeature_12channel(**kwargs):  # L_2
    return Moyun(depth=24,
                 in_channels=12,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)


def test_models_feature_12channel(**kwargs):  # L_2
    return Moyun(depth=24,
                 in_channels=12,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=True,
                 **kwargs)




def moyun_12channel(**kwargs):  # L_2
    return Moyun(depth=24,
                 in_channels=12,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)

def moyun_4channel(**kwargs):  # L_2
    return Moyun(depth=24,
                 in_channels=4,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=16,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)


def moyun_12channel_B(**kwargs):  # L_2
    return Moyun(depth=12,
                 in_channels=12,
                 hidden_size=1024,
                 patch_size=2,
                 num_heads=8,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)

def moyun_4channel_moyun(**kwargs):  # 和原版moyun1保持一致，
    return Moyun(depth=4,
                 in_channels=4,
                 hidden_size=512,
                 patch_size=8,
                 num_heads=8,
                 repa_dep=1,
                 use_kan=False,
                 use_mamba=False,
                 use_stroke=False,
                 use_feature=False,
                 **kwargs)
DiT_models = {
    'test-model': test_models,
    'test-model-nofeature-4channel': test_models_nofeature,
    'test-models-nofeature-12channel': test_models_nofeature_12channel,
    'moyun-12channel': moyun_12channel,
    'moyun-4channel': moyun_4channel,
    "moyun-12channel-B" : moyun_12channel_B,
    "moyun-4channel-cmp1" : moyun_4channel_moyun
    # 'test-models-feature-12channel': test_models_feature_12channel,
    # 'moyun-L-2': moyun_L_2
}

if __name__ == "__main__":
    from torchinfo import summary

    model = moyun_12channel(device='cuda:0').to('cuda')

    # Assuming the shapes of x, y, z
    n = 1
    x = torch.randn(n, 12, 32, 32).to('cuda:0')  # 输入图像
    y = torch.Tensor([(0, 0, 0)] * n).long().to('cuda:0')  # 类别标签
    # print(y.shape)
    # y = torch.rand([n, 3072, 16, 16]).to('cuda:0') # 使用feature
    t = torch.Tensor([1] * n).long().to('cuda:0')  # 时间步

    stroke = [get_ch_strokes_tensor('既').numpy()] * 8
    stroke = np.array(stroke)
    stroke = Tensor(stroke).to('cuda:0')

    model.eval()
    # TODO: stroke channel
    output, _ = model.forward(x, t, y, stroke)
    # print(output)
    summary(model,
            input_data=[x, t, y, stroke],
            col_names=["input_size",
                       "output_size",
                       "num_params",
                       "trainable",
                       "kernel_size",
                       "mult_adds"])
