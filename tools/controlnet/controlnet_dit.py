# -*- coding: utf-8 -*-
"""
controlnet_dit.py — ControlNet for latent DiT (DiT_2Cond-S/2, latent 4×32×32).

设计:
  - 主模型冻结, 只训练 control 分支 (ctrl_encoder + zero_convs)
  - 条件: 3px skel (1通道, 256×256 二值图), 下采样到 32×32 后 patch embed
  - Control 分支: 12 层 DiTBlockSimple (与主模型同 hidden/depth/heads)
  - 注入: x_{i+1} = Block_i(x_i, c) + Z_i(ctrl_i), Z_i zero-init → 完美 warm-start
  - CFG: skel 条件始终提供 (不 drop), CFG 只作用于 callig/char

数学:
    ctrl = CtrlEncoder(skel↓32, c)     # list[L] of (N,256,384)
    for i in range(L):
        x = MainBlock_i(x, c) + zero_proj_i(ctrl[i])
    output = FinalLayer(x) → unpatchify → (N, 8, 32, 32)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# 最小可训练子模块 (与 models.py 对齐; 不 import models.py, 保持解耦)
# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        x = self.proj(x)
        return x


class DiTBlockSimple(nn.Module):
    """与 models.DiTBlock 等价的最小实现 (adaLN-Zero)。"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchEmbed(nn.Module):
    """(N,C,H,W) -> (N,T,D): patch conv + flatten + pos_embed."""
    def __init__(self, patch_size, in_channels, hidden_size, spatial_size=32):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        T = (spatial_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, T, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed


def zero_init_linear(in_f, out_f):
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


# ---------------------------------------------------------------------------
# ControlNet 条件编码器: skel(1,256,256) → 下采样 32×32 → patch embed → 12层 control features
# ---------------------------------------------------------------------------
class ControlConditionEncoder(nn.Module):
    """
    skel 256×256 (1ch) → area downsample 32×32 → PatchEmbed(patch=2) → (N,256,D)
    → 12 × DiTBlockSimple → 逐层 zero_proj 输出 control features
    """
    def __init__(self, in_channels=1, hidden_size=384, depth=12, num_heads=6,
                 cond_spatial=32, cond_patch=2):
        super().__init__()
        self.cond_spatial = cond_spatial
        self.embed = PatchEmbed(cond_patch, in_channels, hidden_size, spatial_size=cond_spatial)
        self.ctrl_blocks = nn.ModuleList([
            DiTBlockSimple(hidden_size, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.out_projs = nn.ModuleList([
            zero_init_linear(hidden_size, hidden_size) for _ in range(depth)
        ])

    def forward(self, cond, c):
        """
        cond: (N, 1, 256, 256) skel 二值图 (0/1)
        c:    (N, D) 主干条件向量 (t_emb + y_emb)
        returns: list[L] of (N, T, D)
        """
        if cond.shape[-1] != self.cond_spatial:
            cond = F.interpolate(cond, size=self.cond_spatial, mode="area")
        x = self.embed(cond)       # (N, T, D)
        feats = []
        for blk, proj in zip(self.ctrl_blocks, self.out_projs):
            x = blk(x, c)
            feats.append(proj(x))  # zero-init → 初始 0
        return feats


# ---------------------------------------------------------------------------
# ControlNet 包装: 给已训练 DiT_2Cond 加注入
# ---------------------------------------------------------------------------
class ControlNetDiT(nn.Module):
    """
    包装已训练的 DiT_2Cond (latent), 注入 skel 结构条件。

    forward(x, t, y_callig, y_char, cond=None):
      cond: (N, 1, 256, 256) skel 二值图; None=无条件 (退化为主模型)
    """
    def __init__(self, main_model, cond_in_channels=1, train_ctrl_only=True):
        super().__init__()
        self.main = main_model
        m = main_model
        hd = m.hidden_size if hasattr(m, "hidden_size") else 384
        depth = len(m.blocks)
        heads = m.num_heads if hasattr(m, "num_heads") else 6

        # latent DiT: input 4×32×32, patch=2 → 256 tokens
        latent_size = m.x_embedder.proj.in_channels  # 4
        spatial = m.x_embedder.proj.kernel_size[0]   # not directly available; infer from pos_embed
        # infer spatial from pos_embed: T = pos_embed.shape[1], patch = sqrt(T)
        T = m.pos_embed.shape[1]
        sp = int(math.sqrt(T)) * m.x_embedder.patch_size[0] if hasattr(m.x_embedder, 'patch_size') else 32
        patch = m.x_embedder.patch_size[0] if hasattr(m.x_embedder, 'patch_size') else 2
        cond_spatial = int(math.sqrt(T)) * patch  # = 32 for latent DiT-S/2

        self.ctrl_encoder = ControlConditionEncoder(
            in_channels=cond_in_channels, hidden_size=hd, depth=depth,
            num_heads=heads, cond_spatial=cond_spatial, cond_patch=patch)

        if train_ctrl_only:
            for p in self.main.parameters():
                p.requires_grad = False

    def forward(self, x, t, y_callig, y_char, cond=None, **kwargs):
        if cond is None:
            return self.main(x, t, y_callig, y_char, **kwargs)

        m = self.main
        # ---- 复现 DiT_2Cond.forward 的 embedding ----
        x = m.x_embedder(x) + m.pos_embed
        if getattr(m, "use_glyph_cond", False) and m.glyph_embedder is not None and kwargs.get("g") is not None:
            g_tok = m.glyph_embedder(kwargs["g"]).flatten(2).transpose(1, 2)
            x = x + m.glyph_scale * g_tok
        t_emb = m.t_embedder(t)

        if m.condition_fusion == "factorized_add":
            e_callig = m.y_callig_embedder(y_callig, False)
            e_char = m.y_char_embedder(y_char, False)
            y_emb = (m.callig_proj(e_callig) + m.char_proj(e_char)) / math.sqrt(2.0)
        elif m.condition_fusion == "xl_highdim":
            e_callig = m.y_callig_embedder(y_callig, False)
            e_char = m.y_char_embedder(y_char, False)
            y_emb = m.cond_fusion(torch.cat([e_callig, e_char], dim=-1)) * m.y_scale
        else:
            e_callig = m.y_callig_embedder(y_callig, False)
            e_char = m.y_char_embedder(y_char, False)
            y_emb = m.cond_fusion(torch.cat([e_callig, e_char], dim=-1))
        c = t_emb + y_emb

        # ---- Control 特征 (逐层) ----
        ctrl_feats = self.ctrl_encoder(cond, c)

        # ---- 主 blocks + 注入 ----
        for i, block in enumerate(m.blocks):
            x = block(x, c)
            if i < len(ctrl_feats):
                x = x + ctrl_feats[i]

        x = m.final_layer(x, c)
        x = m.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, cond=None, **kw):
        """
        CFG 采样: skel 始终提供, callig/char 有/无各跑一遍。
        cond: (N, 1, 256, 256) skel, 对两半都提供 (不 drop)。
        """
        if cond is None:
            return self.main.forward_with_cfg(x, t, y_callig, y_char, cfg_scale=cfg_scale, **kw)
        original_bs = x.shape[0]
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y_callig = torch.cat([y_callig, y_callig], dim=0)
        y_char = torch.cat([y_char, y_char], dim=0)
        uncond_callig = torch.full_like(y_callig, self.main.y_callig_embedder.num_classes)
        uncond_char = torch.full_like(y_char, self.main.y_char_embedder.num_classes)
        y_callig_combined = torch.cat([y_callig[:original_bs], uncond_callig[original_bs:]], dim=0)
        y_char_combined = torch.cat([y_char[:original_bs], uncond_char[original_bs:]], dim=0)
        cond2 = torch.cat([cond, cond], dim=0)  # skel 对两半都提供
        model_out = self.forward(x, t, y_callig_combined, y_char_combined, cond=cond2, **kw)
        model_out = model_out.to(torch.float32)
        eps, rest = model_out[:, :self.main.in_channels], model_out[:, self.main.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        out = torch.cat([eps, rest], dim=1)
        return out[:original_bs]


def load_main_model(model_name="DiT-2Cond-S/2", ckpt_path=None, device="cpu",
                    num_calligraphers=1011, num_characters=35130):
    """加载已训练主模型 (复用 models.py 工厂)。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    from models import DiT_2Cond_models
    model = DiT_2Cond_models[model_name](
        num_calligraphers=num_calligraphers, num_characters=num_characters)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("delta")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device)