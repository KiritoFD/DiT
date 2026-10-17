# -*- coding: utf-8 -*-
"""dit_condition_mixins.py —— **条件系统构造器**（2026-10-14 [重构 PR5] ✓ 大刀阔斧 ✓）

审查 §4.1 原文:
  "当前 `DiT_2Cond.__init__()` 同时负责: Transformer block；书家、书体、字符表；
   STD 编码、条件路由与融合；多种空间风格调制；Skeleton / Deform / ControlNet 遗留功能；
   CFG、初始化及 checkpoint 兼容逻辑。" ✓
  "建议先保留原 `DiT_2Cond` 名称和**权重键**，使其成为**装配器**" ✓
  "⚠ 不能直接按这种新层级替换现有模型：那会改变全部 **state_dict key**" ✗

★ 因此本模块用 **mixin** 拆分 ✓（**不是**新层级 ✗）:
  `class DiT_2Cond(DitConditionMixin, nn.Module)` ✓
  · 方法仍绑定在**同一实例**上 ✓
  · `self.xxx` 属性名**一字未改** ✓ ⇒ **state_dict 键逐字不变** ✓✓
  · 全部 `nn.Module` 参数注册仍留在 `DiT_2Cond.__init__` ✓（本刀只搬方法 ✓）
  ⇒ 等价性由 **前向黄金 + 梯度黄金** 裁决 ✓（二者都是逐位 ✓）

★ 本模块内容 = **逐字搬移** ✓ 的四个构造器 ✓:
  `_build_glyph_condition` / `_build_style_hierarchy` / `_build_deform` / `_build_cond_router`
"""
from __future__ import annotations


import torch
import torch.nn as nn
from src.model.components import get_2d_sincos_pos_embed
from src.model.components import LocalStyleGlyphAdapter
from src.model.components import LowRankSpatialStyleFiLM
from src.model.components import SpatialStyleFiLM
from src.model.components import ScriptGlyphFiLM
from src.model.components import StyleHierarchy
from src.model.components import ZeroCrossAttention
from src.model.components import GlyphStyleCrossAttn
from .glyph_query import GlyphQuery


class DitConditionMixin:
    """条件系统 / 风格层级 / 形变 / 路由的**构造器** ✓（逐字搬移 ✓ 无状态 ✓）。"""

    def _build_glyph_condition(self, *, hidden_size, depth, num_heads, attn_tau,
                               glyph_in_channels, glyph_inject_layers, glyph_inject_mode, glyph_inject_at,
                               glyph_drop_prob, glyph_embedder_depth, glyph_embedder_sep, xattn_q_pos):
        """标准字形（std-soft / 骨架）条件的构造 —— 2026-10-13 从 __init__ 抽出 ✓

        内容与抽出的原文**逐字相同**, 只整体缩进一级 ⇒ 行为不变 ✓（已用
        strict=True 加载真实 ckpt 验证: 0 missing / 0 unexpected ✓）。
        """
        # 甲2 标准字形条件的 token-add 缩放(可学习, 初始 glyph_scale_init, 让字形条件有存在感)
        self.glyph_scale = nn.Parameter(torch.tensor(self.glyph_scale_init))
        if self.glyph_scale_bound > 0:
            if not 0 < self.glyph_scale_init < self.glyph_scale_bound:
                raise ValueError('Bounded glyph scale requires0 < init < bound')
            with torch.no_grad():
                self.glyph_scale.copy_(torch.atanh(torch.tensor(self.glyph_scale_init/self.glyph_scale_bound)))
        # 标准字形条件：独立可训练 glyph_embedder。旧路径编码 G(4,32,32)，
        # std-soft 路径编码 raw mask(1,64,64)，两者都输出与 x 同形的 16x16 token。
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
            if self.std_soft_cond:
                if int(glyph_in_channels) != 1:
                    raise ValueError("std_soft_cond requires glyph_in_channels=1")
                if self.glyph_concat_input:
                    raise ValueError("std_soft_cond is incompatible with glyph_concat_input")
                self.glyph_embedder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=self.std_soft_bias),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=self.std_soft_bias),
                    nn.SiLU(),
                    nn.Conv2d(64, hidden_size, kernel_size=1, bias=False),
                )
            elif self.glyph_embedder_depth <= 0:
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
                # ★ 2026-10-05: 优先用**显式层列表** glyph_inject_at="2,4,6,8"。
                #   空 -> 旧的均匀分布 (depth=12,n=4 -> [2,5,8,11])。
                _at_str = str(glyph_inject_at or "").strip()
                if _at_str:
                    _lst = sorted({int(v) for v in _at_str.split(",") if v.strip() != ""})
                    _lst = [i for i in _lst if 0 <= i < depth]
                    if not _lst:
                        raise ValueError(f"glyph_inject_at 解析为空或全越界: "
                                         f"{glyph_inject_at!r} (depth={depth})")
                    self.glyph_inject_at = _lst
                    n_inj = len(_lst)
                else:
                    self.glyph_inject_at = sorted(
                        set(int(round((i + 1) * depth / n_inj)) - 1 for i in range(n_inj)))
                print(f"[glyph-inject] mode={self.glyph_inject_mode} "
                      f"layers={self.glyph_inject_at} (depth={depth}, "
                      f"{'显式 glyph_inject_at' if _at_str else '均匀分布'})")
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
                            GlyphStyleCrossAttn(hidden_size, num_heads=num_heads,
                                                attn_tau=attn_tau)
                            for _ in self.glyph_inject_at
                        ])
                    else:
                        self.glyph_injections = nn.ModuleList([
                            ZeroCrossAttention(hidden_size, num_heads=num_heads,
                                               grid_size=self.x_embedder.num_patches ** 0.5,
                                               q_pos=bool(xattn_q_pos),
                                               attn_tau=attn_tau)
                            for _ in self.glyph_inject_at
                        ])
                else:
                    # ZeroAdaLNInjection 现居 legacy/controlnet.py (ControlNet 线归档时
                    # 迁入), 但它仍是 adaln 注入的**活跃实现** (v13/v14/v15 全在用)。
                    from .injections import ZeroAdaLNInjection
                    self.glyph_injections = nn.ModuleList([
                        ZeroAdaLNInjection(hidden_size, mode="modulate")
                        for _ in self.glyph_inject_at
                    ])

    def _build_cond_router(self, cond_inject_at, cond_inject_scale, depth):
        """三表条件分层路由(TableCondRouter) —— 2026-10-13 从 __init__ 抽出 ✓（内容与原文逐字相同, 行为不变）"""
        # ── [2026-10-06] 三表条件的分层路由（与 glyph/skel 分支无关）───────────────
        # v54 把表条件 c = t_emb + y_emb 喂给**全部** block；这里改成白名单：
        # 只有 cond_inject_at 列出的 block 允许条件进入，其余层 c = t_emb（完全切断）。
        self.cond_router = None
        _cat = str(cond_inject_at or "").strip()
        if _cat:
            from .injections import TableCondRouter
            try:
                _cdepth = int(depth)
            except Exception:
                _cdepth = len(self.blocks)
            _clst = [int(s) for s in _cat.replace(" ", "").split(",") if s != ""]
            self.cond_router = TableCondRouter(
                _clst, _cdepth, learnable_scale=bool(cond_inject_scale))
            print(f"[cond-route] 表条件只在这些层注入: {self.cond_router.layers} "
                  f"(depth={_cdepth}); 其余层条件切断 (c=t_emb); "
                  f"learnable_scale={bool(cond_inject_scale)}")
        else:
            print("[cond-route] cond_inject_at 为空 -> 旧行为: 表条件给全部层 (v54 语义)")

    def _build_style_hierarchy(self, callig_embed_dim, depth, hidden_size, hier_style, local_ca_at, local_ca_heads, local_ca_impl, local_ca_layers, local_ca_q, local_ca_rank, local_ca_window, lowrank_spatial_rank, num_calligraphers, num_pairs, num_scripts, pair_init, pair_residual, script_embed_dim, script_film, spatial_film_rank):
        """S2 三层语义分解 + 脚本 FiLM + 空间/低秩 FiLM + local_ca —— 2026-10-13 从 __init__ 抽出 ✓（内容与原文逐字相同, 行为不变）"""
        # ── S2：三层语义分解 + 局部风格-骨架引导（2026-09-22）─────────────────
        # 全部默认 None/0 = 与旧 ckpt 逐位等价。
        _grid = int(round(self.x_embedder.num_patches ** 0.5))
        _d_style = int(callig_embed_dim or hidden_size)
        self.hier_style = int(hier_style)
        # ⚠ --script-embed-dim 的历史默认是 None（3cond 时由别处兜底），
        #   独立评测端会原样透传 None -> int(None) TypeError。这里兜底为 64。
        self.script_embed_dim = int(script_embed_dim or 64)
        # 局部通路的 e_cond 维度 = [e_style; e_script]
        self.cond_dim = _d_style + (self.script_embed_dim if self.hier_style > 0 else 0)

        self.style_hier = None
        self.script_film = None
        self.spatial_film = None
        self.lowrank_spatial = None
        self.local_ca = None
        self._local_ca_map = {}
        self.local_ca_q = str(local_ca_q)

        if self.hier_style > 0:
            _n_pair = int(num_pairs) if int(num_pairs) > 0 else int(num_calligraphers)
            self.style_hier = StyleHierarchy(
                num_pairs=_n_pair, num_scripts=int(num_scripts),
                style_dim=_d_style, script_dim=self.script_embed_dim,
                pair_init=str(pair_init), pair_residual=int(pair_residual))
            if bool(script_film):
                self.script_film = ScriptGlyphFiLM(self.script_embed_dim, hidden_size)

        if int(spatial_film_rank) > 0:
            self.spatial_film = SpatialStyleFiLM(
                self.cond_dim, hidden_size, rank=int(spatial_film_rank))

        if int(lowrank_spatial_rank) > 0:
            # 风格侧只用 e_callig (callig_embed_dim), 不拼书体。
            # 书体已经烘进 g 本身, 再乘一次是同一个信号喂两遍。
            self.lowrank_spatial = LowRankSpatialStyleFiLM(
                _d_style, hidden_size, rank=int(lowrank_spatial_rank))

        # 局部通路的位置编码：x 与 g_tok 同为 grid×grid，共用一份 sincos。
        # ⚠ 必须 **任一局部通路开启** 就注册 —— 否则
        #   "只开 spatial_film_rank 不开 local_ca_layers"（配置 s2d_spfilm）
        #   会在 forward 里 AttributeError: local_pos。
        if (int(spatial_film_rank) > 0 or int(local_ca_layers) > 0
                or int(lowrank_spatial_rank) > 0):
            # ⚠ 用独立名字 local_pos，避开 ctx_pos_g 的"属性已存在"register_buffer 坑
            _pe = get_2d_sincos_pos_embed(hidden_size, _grid)
            if "local_pos" in self.__dict__:
                del self.__dict__["local_pos"]
            self.register_buffer("local_pos",
                                 torch.from_numpy(_pe).float().unsqueeze(0),
                                 persistent=False)
        else:
            self.local_pos = None

        if int(local_ca_layers) > 0:
            n_lc = min(int(local_ca_layers), depth)
            if local_ca_at:
                _at = sorted({int(v) for v in str(local_ca_at).split(",")
                              if str(v).strip() != ""})
                _at = [i for i in _at if 0 <= i < depth]
                if not _at:
                    raise ValueError(f"--local-ca-at 解析为空或越界: {local_ca_at!r}")
            else:
                # 均匀分布在 depth 层里（与 glyph_inject_at 同规则）
                _at = sorted(set(int(round((i + 1) * depth / n_lc)) - 1
                                 for i in range(n_lc)))
            self._local_ca_map = {blk: k for k, blk in enumerate(_at)}
            self.local_ca_at = _at
            _lc_cls = (LocalStyleGlyphAdapter
                       if str(local_ca_impl).lower() == "legacy" else GlyphQuery)
            self.local_ca = nn.ModuleList([
                _lc_cls(
                    hidden_size, self.cond_dim,
                    num_heads=int(local_ca_heads), rank=int(local_ca_rank),
                    window=int(local_ca_window), grid_size=_grid)
                for _ in _at
            ])
        else:
            self.local_ca_at = []

        self.initialize_weights()
        if self.std_soft_cond and self.std_soft_zero_init:
            # Step 0 is exactly the base model. The output projection itself
            # receives gradients immediately; earlier convs start once it moves.
            nn.init.zeros_(self.glyph_embedder[-1].weight)
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

    def _build_deform(self, deform_attn_heads, deform_ckpt, deform_coarse, deform_dt_ch, deform_film_mode, deform_grid, deform_max_off, deform_preserve_amp, deform_skel, deform_style_ch, deform_style_tokens, deform_topo, deform_warp_iters, deform_width, gate_radius, res_cap, residual, stroke_cap, stroke_mod):
        """DeformSkel + 残差/笔画调制头 —— 2026-10-13 从 __init__ 抽出 ✓（内容与原文逐字相同, 行为不变）"""
        self.deform_skel = None
        if int(deform_skel) > 0:
            from .deform_skel import DeformSkel
            self.deform_skel = DeformSkel(
                cond_dim=self.cond_dim, ch=4, grid=int(deform_grid),
                style_ch=int(deform_style_ch), width=int(deform_width),
                max_off=float(deform_max_off), coarse=int(deform_coarse),
                residual=int(residual), res_cap=float(res_cap),
                stroke_mod=int(stroke_mod), stroke_cap=float(stroke_cap),
                gate_radius=float(gate_radius), dt_ch=int(deform_dt_ch),
                topo_mode=int(deform_topo), warp_iters=int(deform_warp_iters),
                film_mode=str(deform_film_mode),
                style_tokens=int(deform_style_tokens),
                attn_heads=int(deform_attn_heads),
                preserve_amp=int(deform_preserve_amp))
            if str(deform_ckpt):

                _sd = torch.load(deform_ckpt, map_location="cpu", weights_only=False)
                _sd = _sd.get("deform", _sd) if isinstance(_sd, dict) else _sd
                _miss, _unexp = self.deform_skel.load_state_dict(_sd, strict=False)
                print(f"[deform] 已载入离线权重 {deform_ckpt} "
                      f"(missing={len(_miss)}, unexpected={len(_unexp)})")
                if _miss or _unexp:
                    print(f"[deform] ⚠ 键不匹配: missing={list(_miss)[:4]} "
                          f"unexpected={list(_unexp)[:4]} -> 形变头可能没真正载入")
            print(f"[deform] enabled: width={int(deform_width)} "
                  f"max_off={float(deform_max_off)} coarse={int(deform_coarse)} "
                  f"params={sum(p.numel() for p in self.deform_skel.parameters()):,}")

