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
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


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
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


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


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
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
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
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
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

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

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # Duplicate every sample: first copy conditional, second copy unconditional.
        original_bs = x.shape[0]
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y = torch.cat([y, y], dim=0)
        uncond_y = torch.full_like(y, self.y_embedder.num_classes)
        y_combined = torch.cat([y[:original_bs], uncond_y[original_bs:]], dim=0)
        model_out = self.forward(x, t, y_combined)
        # Apply classifier-free guidance on all learned channels (eps subspace).
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        out = torch.cat([eps, rest], dim=1)
        return out[:original_bs]


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

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


#################################################################################
#                  3-Condition Guided DiT Model (DiT_3Cond)                     #
#################################################################################

class DiT_3Cond(nn.Module):
    """
    Diffusion model with Transformer backbone guided by 3 conditions:
    1. Calligrapher (y_callig)
    2. Script Style (y_script)
    3. Character Content (y_char)
    Also supports intermediate feature extraction for REPA (Representation Alignment) loss.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_calligraphers=142,
        num_scripts=9,
        num_characters=5568,
        learn_sigma=True,
        use_checkpoint=False,
        condition_fusion="legacy",
        callig_embed_dim=None,
        script_embed_dim=None,
        char_embed_dim=None,
        cond_drop_all_prob=0.05,
        cond_drop_one_prob=0.0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition_fusion = condition_fusion
        self.cond_drop_all_prob = float(cond_drop_all_prob)
        self.cond_drop_one_prob = float(cond_drop_one_prob)
        if self.cond_drop_all_prob < 0 or self.cond_drop_one_prob < 0:
            raise ValueError("condition dropout probabilities must be non-negative")
        if self.cond_drop_all_prob + self.cond_drop_one_prob > 1:
            raise ValueError("cond_drop_all_prob + cond_drop_one_prob must be <= 1")

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if condition_fusion == "legacy":
            # Backward-compatible joint MLP used by existing checkpoints.
            self.y_callig_embedder = LabelEmbedder(num_calligraphers, hidden_size, class_dropout_prob)
            self.y_script_embedder = LabelEmbedder(num_scripts, hidden_size, class_dropout_prob)
            self.y_char_embedder = LabelEmbedder(num_characters, hidden_size, class_dropout_prob)
            self.cond_fusion = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.callig_proj = self.script_proj = self.char_proj = None
        elif condition_fusion == "factorized_add":
            # Compact factor tables regularize long-tail identities. Each factor is
            # projected independently and only then added, so an unseen triple uses
            # three transformations that were each trained on other combinations.
            callig_embed_dim = callig_embed_dim or hidden_size
            script_embed_dim = script_embed_dim or hidden_size
            char_embed_dim = char_embed_dim or hidden_size
            self.y_callig_embedder = LabelEmbedder(
                num_calligraphers, callig_embed_dim, 0.0, use_cfg_embedding=True)
            self.y_script_embedder = LabelEmbedder(
                num_scripts, script_embed_dim, 0.0, use_cfg_embedding=True)
            self.y_char_embedder = LabelEmbedder(
                num_characters, char_embed_dim, 0.0, use_cfg_embedding=True)
            self.callig_proj = nn.Sequential(nn.LayerNorm(callig_embed_dim),
                                             nn.Linear(callig_embed_dim, hidden_size))
            self.script_proj = nn.Sequential(nn.LayerNorm(script_embed_dim),
                                             nn.Linear(script_embed_dim, hidden_size))
            self.char_proj = nn.Sequential(nn.LayerNorm(char_embed_dim),
                                           nn.Linear(char_embed_dim, hidden_size))
            self.cond_fusion = None
        else:
            raise ValueError(f"Unknown condition_fusion={condition_fusion!r}")

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

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
        nn.init.normal_(self.y_script_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.y_char_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y_callig, y_script, y_char, return_intermediate_layer=None):
        """
        Forward pass of DiT_3Cond.
        x: (N, C, H, W) noisy latents
        t: (N,) timesteps
        y_callig, y_script, y_char: (N,) condition IDs
        return_intermediate_layer: int index (e.g. 8) to return intermediate patch features for REPA
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t_emb = self.t_embedder(t)               # (N, D)

        if self.condition_fusion == "factorized_add":
            # Controlled masks: either all conditions are dropped for CFG, or
            # exactly one factor is dropped for factor learning. We intentionally
            # avoid accidental two-factor dropout and know the exact full-triple rate.
            if self.training and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0):
                r = torch.rand(y_callig.shape[0], device=y_callig.device)
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which = torch.randint(0, 3, y_callig.shape, device=y_callig.device)
                y_callig = torch.where(drop_all | (drop_one & (which == 0)),
                                       self.y_callig_embedder.num_classes, y_callig)
                y_script = torch.where(drop_all | (drop_one & (which == 1)),
                                       self.y_script_embedder.num_classes, y_script)
                y_char = torch.where(drop_all | (drop_one & (which == 2)),
                                     self.y_char_embedder.num_classes, y_char)

            # Embedders do not perform a second hidden random dropout in this mode.
            e_callig = self.y_callig_embedder(y_callig, False)
            e_script = self.y_script_embedder(y_script, False)
            e_char = self.y_char_embedder(y_char, False)
            y_emb = (self.callig_proj(e_callig)
                     + self.script_proj(e_script)
                     + self.char_proj(e_char)) / math.sqrt(3.0)
        else:
            # Preserve legacy checkpoint behavior: 5% joint drop plus independent
            # LabelEmbedder dropout (normally 10% per factor).
            if self.training:
                drop_all = torch.rand(y_callig.shape[0], device=y_callig.device) < 0.05
                y_callig = torch.where(drop_all, self.y_callig_embedder.num_classes, y_callig)
                y_script = torch.where(drop_all, self.y_script_embedder.num_classes, y_script)
                y_char = torch.where(drop_all, self.y_char_embedder.num_classes, y_char)
            e_callig = self.y_callig_embedder(y_callig, self.training)
            e_script = self.y_script_embedder(y_script, self.training)
            e_char = self.y_char_embedder(y_char, self.training)
            y_emb = self.cond_fusion(torch.cat([e_callig, e_script, e_char], dim=-1))
        c = t_emb + y_emb                        # (N, D)

        intermediate_feats = None
        if self.use_checkpoint:
            for i, block in enumerate(self.blocks):
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    # Run this single block eagerly so its output can be captured for REPA.
                    x = block(x, c)
                    intermediate_feats = x
                else:
                    x = checkpoint(lambda *a: block(*a), x, c, use_reentrant=False)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, c)
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    intermediate_feats = x

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if return_intermediate_layer is not None:
            return x, intermediate_feats
        return x

    def forward_with_cfg(self, x, t, y_callig, y_script, y_char, cfg_scale=4.0):
        """
        Forward pass with Classifier-Free Guidance (CFG).

        Every input sample is duplicated: the first copy runs with its real
        condition ids, the second copy runs with unconditional ids (num_classes).
        Works for arbitrary batch size (1 or >1).
        """
        original_bs = x.shape[0]
        x = torch.cat([x, x], dim=0)                    # (2B, ...)
        t = torch.cat([t, t], dim=0)                    # (2B,)
        y_callig = torch.cat([y_callig, y_callig], dim=0)
        y_script = torch.cat([y_script, y_script], dim=0)
        y_char = torch.cat([y_char, y_char], dim=0)

        # Unconditional ids for the second half (drop IDs).
        uncond_callig = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
        uncond_script = torch.full_like(y_script, self.y_script_embedder.num_classes)
        uncond_char = torch.full_like(y_char, self.y_char_embedder.num_classes)

        # First B rows: conditional; second B rows: unconditional.
        y_callig_combined = torch.cat([y_callig[:original_bs], uncond_callig[original_bs:]], dim=0)
        y_script_combined = torch.cat([y_script[:original_bs], uncond_script[original_bs:]], dim=0)
        y_char_combined = torch.cat([y_char[:original_bs], uncond_char[original_bs:]], dim=0)

        model_out = self.forward(x, t, y_callig_combined, y_script_combined, y_char_combined)
        # Apply CFG on all learned channels (eps subspace), not a hard-coded prefix.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        out = torch.cat([eps, rest], dim=1)
        return out[:original_bs]


def DiT_3Cond_S_2(**kwargs):
    return DiT_3Cond(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_3Cond_B_2(**kwargs):
    return DiT_3Cond(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_3Cond_L_2(**kwargs):
    return DiT_3Cond(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# XL 尺寸：body 维度与官方 DiT-XL-2-256x256.pt 完全一致 (depth=28, hidden=1152, heads=16)，
# 从而能用官方预训练权重加载 transformer body，再在其上做 LoRA 微调。
def DiT_3Cond_XL_2(**kwargs):
    return DiT_3Cond(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


DiT_3Cond_models = {
    'DiT-3Cond-S/2': DiT_3Cond_S_2,
    'DiT-3Cond-B/2': DiT_3Cond_B_2,
    'DiT-3Cond-L/2': DiT_3Cond_L_2,
    'DiT-3Cond-XL/2': DiT_3Cond_XL_2,
}


#################################################################################
#                2-Condition DiT (Calligrapher + Character)                     #
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
        skel_head_enabled=False,
        use_glyph_cond=False,
        glyph_scale_init=0.4,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.condition_fusion = condition_fusion
        self.cond_drop_all_prob = float(cond_drop_all_prob)
        self.cond_drop_one_prob = float(cond_drop_one_prob)
        self.skel_head_enabled = bool(skel_head_enabled)
        self.use_glyph_cond = bool(use_glyph_cond)
        self.glyph_scale_init = float(glyph_scale_init)
        if self.cond_drop_all_prob < 0 or self.cond_drop_one_prob < 0:
            raise ValueError("condition dropout probabilities must be non-negative")
        if self.cond_drop_all_prob + self.cond_drop_one_prob > 1:
            raise ValueError("cond_drop_all_prob + cond_drop_one_prob must be <= 1")

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if condition_fusion == "factorized_add":
            # 二因子可组合条件（V3-A）：calligrapher（风格）× glyph（内容=script×char 合并类）。
            # 每个因子独立低维 embedding + 独立投影后相加，未见的 (callig, glyph) 组合
            # 由两个各自训练充分的边际 score 组合而成，而不是靠一整张联合表 memorization。
            callig_embed_dim = callig_embed_dim or hidden_size
            char_embed_dim = char_embed_dim or hidden_size
            self.y_callig_embedder = LabelEmbedder(
                num_calligraphers, callig_embed_dim, 0.0, use_cfg_embedding=True)
            self.y_char_embedder = LabelEmbedder(
                num_characters, char_embed_dim, 0.0, use_cfg_embedding=True)
            self.callig_proj = nn.Sequential(nn.LayerNorm(callig_embed_dim),
                                             nn.Linear(callig_embed_dim, hidden_size))
            self.char_proj = nn.Sequential(nn.LayerNorm(char_embed_dim),
                                           nn.Linear(char_embed_dim, hidden_size))
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
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # 骨架辅助头（训练引导用，推理不用）：从 final_layer 前的 block 特征
        # 并行解出 1×32×32 latent 骨架预测，与 GT latent 骨架对齐。
        self.skel_head = None
        if self.skel_head_enabled:
            self.skel_head = nn.Sequential(
                nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                nn.Linear(hidden_size, patch_size * patch_size, bias=True),
            )
        # 甲2 标准字形条件的 token-add 缩放(可学习, 初始 glyph_scale_init, 让字形条件有存在感)
        self.glyph_scale = nn.Parameter(torch.tensor(self.glyph_scale_init))
        # 甲2 标准字形条件：独立可训练 glyph_embedder(Conv2d 4→hidden, patch 编码)
        # 把标准字形 latent G(4,32,32) 编成与 x token 同形 token, forward 时 token-add。
        # 独立投影而非复用 x_embedder: 保证 g 编码可学习、norm 可控, 不依赖 x_embedder
        # 是否被冻结(XL LoRA 模式下 x_embedder 冻结, 复用会导致 g 信号被锁死)。
        self.glyph_embedder = None
        if self.use_glyph_cond:
            ps_ = self.x_embedder.patch_size[0] if not isinstance(self.x_embedder.patch_size, int) else self.x_embedder.patch_size
            self.glyph_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=ps_, stride=ps_, bias=False)
        self.initialize_weights()

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
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
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
            if self.training and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0):
                r = torch.rand(y_callig.shape[0], device=y_callig.device)
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which = torch.randint(0, 2, y_callig.shape, device=y_callig.device)
                y_callig = torch.where(drop_all | (drop_one & (which == 0)),
                                       self.y_callig_embedder.num_classes, y_callig)
                y_char = torch.where(drop_all | (drop_one & (which == 1)),
                                     self.y_char_embedder.num_classes, y_char)
            e_callig = self.y_callig_embedder(y_callig, False)
            e_char = self.y_char_embedder(y_char, False)
            y_emb = (self.callig_proj(e_callig) + self.char_proj(e_char)) / math.sqrt(2.0)
        elif self.condition_fusion == "xl_highdim":
            # XL 高维条件：与 factorized_add 相同的 4-way 可控 mask（CFG 需要 uncond 维度）。
            if self.training and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0):
                r = torch.rand(y_callig.shape[0], device=y_callig.device)
                drop_all = r < self.cond_drop_all_prob
                drop_one = ((r >= self.cond_drop_all_prob)
                            & (r < self.cond_drop_all_prob + self.cond_drop_one_prob))
                which = torch.randint(0, 2, y_callig.shape, device=y_callig.device)
                y_callig = torch.where(drop_all | (drop_one & (which == 0)),
                                       self.y_callig_embedder.num_classes, y_callig)
                y_char = torch.where(drop_all | (drop_one & (which == 1)),
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

        intermediate_feats = None
        if self.use_checkpoint:
            for i, block in enumerate(self.blocks):
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    # Run this single block eagerly so its output can be captured for REPA.
                    x = block(x, c)
                    intermediate_feats = x
                else:
                    x = checkpoint(lambda *a: block(*a), x, c, use_reentrant=False)
        else:
            for i, block in enumerate(self.blocks):
                x = block(x, c)
                if return_intermediate_layer is not None and i == return_intermediate_layer:
                    intermediate_feats = x

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


def DiT_2Cond_S_2(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_2Cond_S_4(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_2Cond_B_2(**kwargs):
    return DiT_2Cond(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_2Cond_XL_2(**kwargs):
    return DiT_2Cond(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


DiT_2Cond_models = {
    'DiT-2Cond-S/2': DiT_2Cond_S_2,
    'DiT-2Cond-S/4': DiT_2Cond_S_4,
    'DiT-2Cond-B/2': DiT_2Cond_B_2,
    'DiT-2Cond-XL/2': DiT_2Cond_XL_2,
}

