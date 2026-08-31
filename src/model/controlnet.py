# -*- coding: utf-8 -*-
"""
controlnet_dit.py — ControlNet for latent DiT (DiT_2Cond-S/2, latent 4×32×32).

设计
----
  - 主模型完全冻结, 只训练 control 分支 (ctrl_encoder + zero_convs)
  - 条件: skel 的 VAE latent (4ch, 32×32), 与主模型 x latent 同空间
  - Control 分支: N 层现代 DiTBlock (RMSNorm / SwiGLU / 2D-RoPE / QK-Norm)
  - 注入: adaLN 式调制 ``x = x * (1 + s) + t``，(s, t) 由 zero-init Linear 产出
    → 训练起始时注入严格为 0，主模型行为不变（完美 warm-start）
  - CFG: skel 条件始终提供 (不 drop), CFG 只作用于 callig/char

与旧实现的差异（本次重构）
--------------------------
1) **架构与主模型对齐（现代化）**
   旧版 ctrl encoder 用自写的 LayerNorm/GELU/无 RoPE 的 attention。
   主模型升级到 RMSNorm+SwiGLU+RoPE 之后，ctrl 侧若仍是旧组件，会导致
   ctrl 特征与残差流的分布/位置语义不匹配。现在两侧共用
   ``src.model.modules``，并由同一组 arch 参数控制。

2) **RoPE 必须透传**
   主模型 ``rope=True`` 时，位置信息在 attention 内部通过 q/k 旋转注入，
   不再加到残差流上。旧代码 ``block(x, c)`` 不传 rope → **主模型在
   ControlNet 包装下会静默丢失位置信息**（训练/推理不一致且极难排查）。
   现在 main block 与 ctrl block 都接收同一份 rope 表。

3) **注入方式：加法 → adaLN 式调制**
   ``x = x + feat`` 只能做平移；``x = x*(1+s) + t`` 让 control 既能增强
   也能抑制主残差流，对骨骼这种"稀疏强结构"条件更合适。
   两者都是 zero-init → 恒等初始化。``injection="add"`` 可退回旧行为。

4) **null condition 不再是零 latent**
   零 latent 经 VAE 解码不是"空白"而是某种特定的灰色块，与真实骨骼分布
   差距很大，导致 CFG 的 uncond 分支落在分布外。
   现在支持 ``null_cond="gaussian"``(默认) / ``"zeros"``(旧) / ``"learned"``。

5) **主模型 ckpt 路径失效时硬失败**
   旧代码 ``os.path.exists(...) or None`` 会静默用随机初始化的主模型训练
   ControlNet —— 几十小时后才从 eval 结果发现问题。现在直接抛异常。

6) **轻量 encoder**
   ``ctrl_depth`` / ``ctrl_hidden`` 允许把 encoder 做浅/做窄。旧版强制复制
   主模型全部 12 层，参数翻倍，对 118k 样本偏重。

关键 infra:
  - forward 时主模型不建图 (requires_grad=False), 只有 ctrl_encoder 建图
  - ctrl_encoder 在 bf16 autocast 外运行 (小模型, fp32 即可, 避免 autocast graph 陷阱)
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import modules as M


# ---------------------------------------------------------------------------
# 注入层：zero-init，起始为恒等
# ---------------------------------------------------------------------------
def zero_init_linear(in_f, out_f):
    lin = nn.Linear(in_f, out_f)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class ZeroAdaLNInjection(nn.Module):
    """adaLN 式零初始化注入：``out = x * (1 + s) + t``。

    ``s``/``t`` 由同一个 zero-init Linear 产出，因此 init 时 s=t=0，
    注入严格为恒等 —— 与 ControlNet 的 zero-conv warm-start 语义一致。

    梯度种子（为什么 step 0 就能学到东西）：
        d(out)/d(W) = x   （ctrl block 输出，非零 → W 立刻有梯度）
        d(out)/d(b) = 1   （bias 立刻有梯度）
        d(out)/d(x) = W = 0 → **ctrl blocks 在 W 变非零前收不到梯度**
    这是 ControlNet 的正确行为（先学注入权重，再学控制特征）。
    """

    def __init__(self, hidden_size, mode="modulate"):
        super().__init__()
        if mode not in ("modulate", "add"):
            raise ValueError(f"Unknown injection mode={mode!r}")
        self.mode = mode
        self.proj = zero_init_linear(hidden_size, hidden_size * (2 if mode == "modulate" else 1))

    def forward(self, x, feat):
        if self.mode == "modulate":
            s, t = self.proj(feat).chunk(2, dim=-1)
            return x * (1.0 + s) + t
        return x + self.proj(feat)


# ---------------------------------------------------------------------------
# ControlNet 条件编码器
# ---------------------------------------------------------------------------
class ControlConditionEncoder(nn.Module):
    """skel latent (4,32,32) → PatchEmbed(p=2) → (N,256,D) → N × DiTBlock → 逐层输出。

    与主模型共用 ``src.model.modules`` 的组件，因此 arch 参数（norm/mlp/rope/qk_norm）
    需要与主模型一致，否则两侧的特征分布会不匹配。
    """

    def __init__(self, in_channels=4, hidden_size=384, depth=12, num_heads=6,
                 cond_spatial=32, cond_patch=2, cond_dim=None,
                 norm_type="rms", mlp_type="swiglu", qk_norm=True,
                 rope=True, rope_theta=100.0, attn_impl="sdpa"):
        super().__init__()
        self.cond_spatial = int(cond_spatial)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        # 时间/语义条件向量 c 由主模型产出（维度 = 主模型 hidden_size）。
        # ctrl encoder 更窄时需要投影，否则 adaLN 的 Linear 维度对不上。
        self.c_proj = (nn.Linear(int(cond_dim), hidden_size)
                       if cond_dim and int(cond_dim) != hidden_size else nn.Identity())

        num_patches = (self.cond_spatial // cond_patch) ** 2
        head_dim = hidden_size // num_heads
        if head_dim * num_heads != hidden_size:
            raise ValueError(
                f"ctrl_hidden_size({hidden_size}) must be divisible by num_heads({num_heads})")

        # 独立的 patch embed（输入通道与主模型不同：skel latent 4ch）
        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=cond_patch, stride=cond_patch)
        # rope=True 时位置由 RoPE 提供，不加 pos_embed；rope=False 时保留（旧行为）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.rope = bool(rope)

        self.ctrl_blocks = nn.ModuleList([
            M.DiTBlock(hidden_size, num_heads, mlp_ratio=4.0,
                       norm_type=norm_type, mlp_type=mlp_type,
                       qk_norm=qk_norm, attn_impl=attn_impl)
            for _ in range(depth)
        ])

        grid = int(round(num_patches ** 0.5))
        assert grid * grid == num_patches, "ctrl RoPE 只支持正方形 token grid"
        if self.rope:
            cos, sin = M.precompute_rope_2d(grid, head_dim, theta=rope_theta)
            self.register_buffer("rope_cos", cos.float(), persistent=False)
            self.register_buffer("rope_sin", sin.float(), persistent=False)
        else:
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)

    @property
    def rope_pair(self):
        return (self.rope_cos, self.rope_sin) if self.rope else None

    def forward(self, cond, c):
        """
        cond: (N, 4, 32, 32) skel VAE latent
        c:    (N, D) 主干条件向量 (t_emb + y_emb)
        returns: list[depth] of (N, T, D)（未经 zero_proj，投影在 ControlNetDiT 里做）
        """
        if cond.shape[-1] != self.cond_spatial:
            cond = F.interpolate(cond, size=self.cond_spatial, mode="bicubic")
        x = self.proj(cond).flatten(2).transpose(1, 2)
        if not self.rope:
            x = x + self.pos_embed
        c = self.c_proj(c)
        rope = self.rope_pair
        feats = []
        for blk in self.ctrl_blocks:
            x = blk(x, c, rope=rope)
            feats.append(x)
        return feats


# ---------------------------------------------------------------------------
# ControlNet 包装
# ---------------------------------------------------------------------------
class ControlNetDiT(nn.Module):
    """包装已训练的 DiT_2Cond (latent), 注入 skel 结构条件。

    forward(x, t, y_callig, y_char, cond=None, **kwargs):
      cond: (N, 4, 32, 32) skel VAE latent; None = 无条件 (退化为主模型)

    关键: 主模型 requires_grad=False, ctrl_encoder requires_grad=True.
    """

    def __init__(self, main_model, cond_in_channels=4, train_ctrl_only=True,
                 ctrl_depth=None, ctrl_hidden=None, ctrl_num_heads=None,
                 injection="modulate", null_cond="gaussian",
                 norm_type=None, mlp_type=None, qk_norm=None,
                 rope=None, rope_theta=None, attn_impl=None):
        """
        Args:
            main_model: 已加载权重的主模型（DiT_2Cond）。
            norm_type/mlp_type/qk_norm/rope/... : ctrl encoder 的架构参数。
                **默认 None = 从 main_model 继承**。若主模型是旧架构
                （LayerNorm + GELU，如 s15~s19 的 ckpt），不继承就会默认走
                新版（RMSNorm + SwiGLU），ctrl_encoder 的 mlp 形状会是
                1024 而 ckpt 里是 1536，在 load_state_dict 时报错。
                只有当你明确想让 ctrl encoder 与主模型用不同架构时才显式传。
        """
        super().__init__()
        self.main = main_model
        m = main_model

        def _inherit(v, attr, default):
            if v is not None:
                return v
            return getattr(m, attr, default)

        norm_type = _inherit(norm_type, 'norm_type', 'rms')
        mlp_type = _inherit(mlp_type, 'mlp_type', 'swiglu')
        qk_norm = _inherit(qk_norm, 'qk_norm', True)
        rope = _inherit(rope, 'rope', True)
        rope_theta = _inherit(rope_theta, 'rope_theta', 100.0)
        attn_impl = _inherit(attn_impl, 'attn_impl', 'sdpa')

        hd = int(getattr(m, 'hidden_size', 384))
        depth = len(m.blocks)
        heads = int(getattr(m, 'num_heads', 6))

        # 轻量 encoder：默认与主模型同深；设为 depth//2 可省一半参数
        _ctrl_depth = int(ctrl_depth) if ctrl_depth else depth
        _ctrl_hidden = int(ctrl_hidden) if ctrl_hidden else hd
        _ctrl_heads = int(ctrl_num_heads) if ctrl_num_heads else heads
        if _ctrl_heads <= 0 or _ctrl_hidden % _ctrl_heads != 0:
            raise ValueError(
                f"ctrl_hidden({_ctrl_hidden}) must be divisible by ctrl_num_heads({_ctrl_heads})")

        T = m.pos_embed.shape[1]
        p = m.x_embedder.patch_size[0] if hasattr(m.x_embedder, 'patch_size') else 2
        cond_spatial = int(math.sqrt(T)) * p  # 32 for DiT-S/2

        self.injection_mode = injection
        self.null_cond = null_cond

        self.ctrl_encoder = ControlConditionEncoder(
            in_channels=cond_in_channels, hidden_size=_ctrl_hidden,
            depth=_ctrl_depth, num_heads=_ctrl_heads,
            cond_spatial=cond_spatial, cond_patch=p, cond_dim=hd,
            norm_type=norm_type, mlp_type=mlp_type, qk_norm=qk_norm,
            rope=rope, rope_theta=rope_theta, attn_impl=attn_impl)

        # 注入目标：主模型的**最后** _ctrl_depth 个 block。
        # ctrl_depth == main depth 时即全部 block（与旧行为一致）。
        self.inject_layers = list(range(depth - _ctrl_depth, depth))

        # ctrl_hidden != main hidden 时用 Linear 对齐通道；否则 1×1（等价于不映射）
        if _ctrl_hidden != hd:
            self.ctrl_to_main = nn.Linear(_ctrl_hidden, hd)
        else:
            self.ctrl_to_main = nn.Identity()

        self.injections = nn.ModuleList([
            ZeroAdaLNInjection(hd, mode=injection) for _ in range(_ctrl_depth)
        ])

        # 可学习的 null condition（null_cond="learned" 时用）
        if null_cond == "learned":
            self.null_cond_param = nn.Parameter(
                torch.zeros(1, cond_in_channels, cond_spatial, cond_spatial))
        else:
            self.register_parameter("null_cond_param", None)

        if train_ctrl_only:
            for param in self.main.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------ #
    # 旧 ckpt 兼容
    # ------------------------------------------------------------------ #
    @staticmethod
    def _remap_legacy_ctrl_keys(sd):
        """把 2026-08-28 之前的旧 ckpt key 映射到当前布局。

        旧布局（controlnet_legacy.py）::

            ctrl_encoder.embed.pos_embed / .proj.weight / .proj.bias
            ctrl_encoder.out_projs.{i}.weight / .bias        # zero-conv
            ctrl_encoder.ctrl_blocks.{i}.*

        新布局::

            ctrl_encoder.pos_embed                            # persistent buffer
            ctrl_encoder.proj.weight / .bias                  # PatchEmbed
            injections.{i}.proj.weight / .bias                # ZeroAdaLNInjection
            ctrl_encoder.ctrl_blocks.{i}.*

        不映射的后果很隐蔽：``load_state_dict(strict=False)`` 不报错，
        但 injections 停留在 zero-init，cond 完全短路 —— 四臂评估结果
        会完全相同，看上去像"ControlNet 无效"而不是"权重没加载"。
        """
        out = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("ctrl_encoder.embed."):
                nk = nk.replace("ctrl_encoder.embed.", "ctrl_encoder.", 1)
            elif nk.startswith("ctrl_encoder.out_projs."):
                # out_projs.{i}.{weight,bias} -> injections.{i}.proj.{weight,bias}
                rest = nk[len("ctrl_encoder.out_projs."):]
                idx, _, leaf = rest.partition(".")
                nk = f"injections.{idx}.proj.{leaf}"
            out[nk] = v
        return out

    @classmethod
    def from_ckpt(cls, main_model, ckpt_path, device=None, strict=True, **kwargs):
        """从 ckpt 构造。自动处理 {ema|ctrl|model} 包装与旧 key 布局。

        strict=True（默认）时，若存在**未加载的 ctrl 参数**会直接抛错 ——
        因为那意味着 injections 停在 zero-init，cond 被完全短路，
        而 strict=False 会让这种失效静默发生。
        """
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck
        if isinstance(ck, dict):
            for key in ("ema", "ctrl", "model", "state_dict"):
                if key in ck and isinstance(ck[key], dict):
                    sd = ck[key]
                    break
        sd = cls._remap_legacy_ctrl_keys(sd)
        obj = cls(main_model, **kwargs)
        if device is not None:
            obj = obj.to(device)
        miss, unexp = obj.load_state_dict(sd, strict=False)
        # main.* 由主模型自己加载，不在这里管
        ctrl_miss = [k for k in miss if not k.startswith("main.")]
        ctrl_unexp = [k for k in unexp if not k.startswith("main.")]
        import logging
        log = logging.getLogger(__name__)
        step = ck.get("train_steps") if isinstance(ck, dict) else "?"
        log.warning(f"[ctrl-ckpt] {ckpt_path} step={step} "
                    f"ctrl_missing={len(ctrl_miss)} ctrl_unexpected={len(ctrl_unexp)}")
        if ctrl_unexp:
            log.warning(f"    unexpected(前5): {ctrl_unexp[:5]}")
        if ctrl_miss and strict:
            raise RuntimeError(
                f"ControlNet ckpt 有 {len(ctrl_miss)} 个 ctrl 参数未加载 "
                f"(例: {ctrl_miss[:3]})。injections 若未加载会因 zero-init 而"
                f"让 cond 完全短路。请检查架构参数是否与训练时一致，"
                f"或显式传 strict=False 以忽略。")
        return obj

    # ---- null condition ----
    def _make_null(self, cond):
        """生成"无骨骼信息"的 cond。

        zeros    : 旧行为 —— 解码后是特定灰色块，与真实骨骼分布差距大
        gaussian : 默认 —— 更接近"无信息"的先验，CFG uncond 分支落在分布内
        learned  : 可学习张量
        """
        if self.null_cond == "learned":
            return self.null_cond_param.expand(cond.shape[0], -1, -1, -1)
        if self.null_cond == "zeros":
            return torch.zeros_like(cond)
        return torch.randn_like(cond)

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

    @property
    def _main_rope(self):
        """主模型的 RoPE 表（rope=False 时为 None）。"""
        m = self.main
        if getattr(m, 'rope', False) and m.rope_cos is not None:
            return (m.rope_cos, m.rope_sin)
        return None

    def forward(self, x, t, y_callig, y_char, cond=None, **kwargs):
        """
        cond: (N,4,32,32) skel VAE latent or None.
        当 cond=None 时退化为主模型 forward (用于训练时条件 dropout).
        """
        if cond is None:
            return self.main(x, t, y_callig, y_char, **kwargs)

        m = self.main
        # ---- 复现 DiT_2Cond.forward 的 embedding ----
        # 注意：主模型 rope=True 时 **不能** 加 pos_embed（位置在 attention 内注入）
        x = m.x_embedder(x)
        if not getattr(m, 'rope', False):
            x = x + m.pos_embed
        if getattr(m, "use_glyph_cond", False) and m.glyph_embedder is not None \
                and kwargs.get("g") is not None:
            g_tok = m.glyph_embedder(kwargs["g"]).flatten(2).transpose(1, 2)
            x = x + m.glyph_scale * g_tok
        t_emb = m.t_embedder(t)
        c = self._compute_condition(m, y_callig, y_char, t_emb, **kwargs)

        # ---- Control 特征（逐层）----
        ctrl_feats = self.ctrl_encoder(cond, c)

        # ---- 主 blocks + 注入 ----
        rope = self._main_rope
        inject = dict(zip(self.inject_layers, self.injections))
        for i, block in enumerate(m.blocks):
            x = block(x, c, rope=rope)          # rope 必须透传，否则主模型丢位置信息
            if i in inject:
                feat = self.ctrl_to_main(ctrl_feats[self.inject_layers.index(i)])
                x = inject[i](x, feat)

        # ---- final ----
        x = m.final_layer(x, c)
        return m.unpatchify(x)

    def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale=4.0, cond=None, **kw):
        """
        CFG 采样: skel 始终提供 (对两半), callig/char 有/无各跑一遍.

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


# ---------------------------------------------------------------------------
# 主模型加载
# ---------------------------------------------------------------------------
def load_main_model(model_name="DiT-2Cond-S/2", ckpt_path=None, device="cpu",
                    num_calligraphers=1011, num_characters=35130,
                    condition_fusion="factorized_add",
                    callig_embed_dim=128, char_embed_dim=256,
                    char_proj_mode="full", freeze_char_table=False,
                    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
                    cond_drop_which_glyph_prob=0.5,
                    use_checkpoint=False, learn_sigma=None,
                    diffusion_type="flow",
                    norm_type="rms", mlp_type="swiglu", qk_norm=True,
                    rope=True, rope_theta=100.0, attn_impl="sdpa",
                    # ---- IDS 组件码本字嵌入 (s25 及以后主模型) ----
                    use_ids_char_embedder=False,
                    ids_file=None, char_id_to_char=None,
                    # ---- 标准字形 DINO 字嵌入 (s28 主模型) ----
                    use_std_dino_char_embedder=False,
                    std_dino_table_path=None,
                    chars_per_script=7026):
    """加载已训练主模型（复用 src.model.dit 工厂）。

    ``ckpt_path`` 非空但文件不存在时**直接抛异常**。
    旧实现的 ``os.path.exists(...) or None`` 会静默用随机初始化的主模型训练
    ControlNet —— 症状是几十小时后 eval 指标始终不涨，极难定位。
    """
    from src.model import DiT_2Cond_models

    # ---- ckpt 硬断言 ----
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[load_main_model] 主模型 ckpt 不存在: {ckpt_path}\n"
                f"  cwd = {os.getcwd()}\n"
                f"  ControlNet 必须在已训练主模型之上训练；"
                f"绝不能静默回退到随机初始化。请检查 --main-ckpt 路径。")
    else:
        raise ValueError(
            "[load_main_model] 未提供 ckpt_path。ControlNet 训练必须显式指定 "
            "--main-ckpt；若确需从随机主模型调试，请显式传 allow_random_main=True。")

    if learn_sigma is None:
        # 与 train.py 的自动规则保持一致：flow 无方差头
        learn_sigma = str(diffusion_type).lower() not in ("flow", "flow_matching", "fm")

    model = DiT_2Cond_models[model_name](
        num_calligraphers=num_calligraphers, num_characters=num_characters,
        condition_fusion=condition_fusion,
        callig_embed_dim=callig_embed_dim, char_embed_dim=char_embed_dim,
        char_proj_mode=char_proj_mode, freeze_char_table=freeze_char_table,
        use_ids_char_embedder=use_ids_char_embedder,
        ids_file=ids_file, char_id_to_char=char_id_to_char,
        use_std_dino_char_embedder=use_std_dino_char_embedder,
        std_dino_table_path=std_dino_table_path,
        chars_per_script=chars_per_script,
        cond_drop_all_prob=cond_drop_all_prob, cond_drop_one_prob=cond_drop_one_prob,
        cond_drop_which_glyph_prob=cond_drop_which_glyph_prob,
        use_checkpoint=use_checkpoint, learn_sigma=learn_sigma,
        norm_type=norm_type, mlp_type=mlp_type, qk_norm=qk_norm,
        rope=rope, rope_theta=rope_theta, attn_impl=attn_impl)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("delta") or ck
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # 把关：区分"架构演进导致的预期缺失"与"真正的加载失败"
    if len(unexpected) and len(missing) == 0 and len(unexpected) <= 4:
        # 例如旧 ckpt 含 skel_head / glyph_embedder，新配置未启用 —— 可接受
        pass
    if len(missing) > 0:
        # pos_embed 是 persistent buffer，旧 ckpt 会有；此处缺失通常是架构不匹配
        print(f"[load][WARN] {len(missing)} missing keys, e.g. {missing[:5]}")

    print(f"[load] {os.path.basename(ckpt_path)} missing={len(missing)} "
          f"unexpected={len(unexpected)} "
          f"(char_embed_dim={char_embed_dim}, char_proj_mode={char_proj_mode}, "
          f"freeze_char_table={freeze_char_table}, learn_sigma={learn_sigma}, "
          f"arch={norm_type}/{mlp_type}/rope={rope})")
    return model.to(device)
