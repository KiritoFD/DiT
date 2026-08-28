# -*- coding: utf-8 -*-
"""
controlnet_dit.py — ControlNet for latent DiT (DiT_2Cond-S/2, latent 4×32×32).

设计:
  - 主模型完全冻结, 只训练 control 分支 (ctrl_encoder + zero_convs)
  - 条件: 3px skel (1通道, 256×256 二值图), area downsample 到 32×32
  - Control 分支: 12 层 DiTBlockSimple (与主模型同 hidden=384 / depth=12 / heads=6)
  - 注入: x_{i+1} = MainBlock_i(x_i, c) + Z_i(ctrl_i), Z_i zero-init → 完美 warm-start
  - CFG: skel 条件始终提供 (不 drop), CFG 只作用于 callig/char

关键 infra:
  - forward 时主模型不建图 (requires_grad=False), 只有 ctrl_encoder 建图
  - ctrl_encoder 在 bf16 autocast 外运行 (小模型, fp32 即可, 避免 autocast graph 陷阱)
  - pred_xstart graph 不被保留 (不用于 struct loss, 只用于 eps loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


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
# ControlNet 条件编码器: skel(1,256,256) → 32×32 → patch embed → 12层 control features
# ---------------------------------------------------------------------------
class ControlConditionEncoder(nn.Module):
    """
    skel 256×256 (1ch) → area downsample 32×32 → PatchEmbed(patch=2) → (N,256,D)
    → 12 × DiTBlockSimple → 逐层 zero_proj 输出 control features

    out_projs 零初始化: 训练开始时 ctrl 注入 = 0, 主模型行为不变 (完美 warm-start).
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
            # proj is zero-init (weight=0, bias=0). At init, feat=0 → no injection.
            # The learning seed: d(out)/d(proj.weight) = x (the ctrl block output,
            # non-zero), so proj.weight gets gradient even at step 0.
            # d(out)/d(proj.bias) = 1, so proj.bias also gets gradient.
            # d(out)/d(x) = proj.weight = 0, so ctrl blocks get NO gradient until
            # proj.weight becomes non-zero. This is correct ControlNet warm-start.
            feats.append(proj(x))
        return feats


# ---------------------------------------------------------------------------
# ControlNet 包装: 给已训练 DiT_2Cond 加注入
# ---------------------------------------------------------------------------
class ControlNetDiT(nn.Module):
    """
    包装已训练的 DiT_2Cond (latent), 注入 skel 结构条件。

    forward(x, t, y_callig, y_char, cond=None, **kwargs):
      cond: (N, 1, 256, 256) skel 二值图; None=无条件 (退化为主模型)

    关键: 主模型 requires_grad=False, ctrl_encoder requires_grad=True.
    forward 时不重新实现主模型 forward (避免与 models.py 漂移), 而是直接调用
    主模型 forward, 然后在 block 之间注入 — 通过 hook 实现.
    """
    def __init__(self, main_model, cond_in_channels=1, train_ctrl_only=True):
        super().__init__()
        self.main = main_model
        m = main_model
        hd = getattr(m, 'hidden_size', 384)
        depth = len(m.blocks)
        heads = getattr(m, 'num_heads', 6)

        # Infer latent spatial size from pos_embed: T tokens, patch=p
        T = m.pos_embed.shape[1]
        p = m.x_embedder.patch_size[0] if hasattr(m.x_embedder, 'patch_size') else 2
        cond_spatial = int(math.sqrt(T)) * p  # 32 for DiT-S/2

        self.ctrl_encoder = ControlConditionEncoder(
            in_channels=cond_in_channels, hidden_size=hd, depth=depth,
            num_heads=heads, cond_spatial=cond_spatial, cond_patch=p)

        if train_ctrl_only:
            for param in self.main.parameters():
                param.requires_grad = False
        # ctrl_encoder params are trainable by default (new module)

    def _compute_condition(self, m, y_callig, y_char, t_emb, **kwargs):
        """复现 DiT_2Cond.forward 的条件融合, 返回 c = t_emb + y_emb."""
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
        return t_emb + y_emb

    def forward(self, x, t, y_callig, y_char, cond=None, **kwargs):
        """
        cond: (N,1,256,256) skel or None.
        当 cond=None 时退化为主模型 forward (用于训练时条件 dropout).
        """
        if cond is None:
            return self.main(x, t, y_callig, y_char, **kwargs)

        m = self.main
        # ---- 复现 DiT_2Cond.forward 的 embedding ----
        x = m.x_embedder(x) + m.pos_embed
        if getattr(m, "use_glyph_cond", False) and m.glyph_embedder is not None and kwargs.get("g") is not None:
            g_tok = m.glyph_embedder(kwargs["g"]).flatten(2).transpose(1, 2)
            x = x + m.glyph_scale * g_tok
        t_emb = m.t_embedder(t)
        c = self._compute_condition(m, y_callig, y_char, t_emb, **kwargs)

        # ---- Control 特征 (逐层, zero-init) ----
        ctrl_feats = self.ctrl_encoder(cond, c)

        # ---- 主 blocks + 注入 ----
        for i, block in enumerate(m.blocks):
            x = block(x, c)
            if i < len(ctrl_feats):
                x = x + ctrl_feats[i]

        # ---- final ----
        x = m.final_layer(x, c)
        x = m.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, cond=None, **kw):
        """
        CFG 采样: skel 始终提供 (对两半), callig/char 有/无各跑一遍.
        cond: (N,1,256,256) skel.
        Model may be DDPM-eps or Flow-velocity: the CFG recombination below
        is applied on the first ``in_channels`` channels in both cases (flow
        drops its sigma channels here, mirroring main.forward_with_cfg).
        Flow sampler must NOT clip the output (velocity is unclipped).
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
        cond2 = torch.cat([cond, cond], dim=0)
        model_out = self.forward(x, t, y_callig_combined, y_char_combined, cond=cond2, **kw)
        model_out = model_out.to(torch.float32)
        eps, rest = model_out[:, :self.main.in_channels], model_out[:, self.main.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        out = torch.cat([eps, rest], dim=1)
        return out[:original_bs]


def load_main_model(model_name="DiT-2Cond-S/2", ckpt_path=None, device="cpu",
                    num_calligraphers=1011, num_characters=35130,
                    condition_fusion="factorized_add",
                    callig_embed_dim=128, char_embed_dim=256,
                    char_proj_mode="full", freeze_char_table=False,
                    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
                    cond_drop_which_glyph_prob=0.5,
                    use_checkpoint=False, learn_sigma=True):
    """加载已训练主模型 (复用 src.model.dit 工厂)。"""
    from src.model import DiT_2Cond_models
    model = DiT_2Cond_models[model_name](
        num_calligraphers=num_calligraphers, num_characters=num_characters,
        condition_fusion=condition_fusion,
        callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
        char_proj_mode=char_proj_mode, freeze_char_table=freeze_char_table,
        cond_drop_all_prob=cond_drop_all_prob, cond_drop_one_prob=cond_drop_one_prob,
        cond_drop_which_glyph_prob=cond_drop_which_glyph_prob,
        use_checkpoint=use_checkpoint, learn_sigma=learn_sigma)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck.get("delta") or ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] {os.path.basename(ckpt_path)} missing={len(missing)} "
              f"unexpected={len(unexpected)} "
              f"(char_embed_dim={char_embed_dim}, char_proj_mode={char_proj_mode}, "
              f"freeze_char_table={freeze_char_table})")
    return model.to(device)
