# -*- coding: utf-8 -*-
"""config.py — DiT 构造参数的**类型化分组**（114 个参数 → 7 组）

动机
----
`DiT_2Cond.__init__` 有 114 个参数，原先被手写三遍（签名 + train + eval），
且按前缀天然分成 6 簇（glyph 14 / deform 14 / cond 11 / callig 8 / style 7 / local 7）。
把它们收进 dataclass 分组后:
  · **新增参数只需加到对应组** —— 训练/评测自动同步 ✓（漏传 -> 架构静默不一致 ✗ 这一类 bug 结构性消失）
  · 每组自带语义与默认值, 可读性大幅提升
  · `build.from_args()` 只做一次 Namespace -> DiTConfig 的映射 ✓

约定: 字段名 == 原来的 kwargs 名 ✓（`to_kwargs()` 原样展开给 `DiT_2Cond(...)` ✓），
      因此模型内部一行都不用改 ✓（零风险 ✓）。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

__all__ = ["CoreCfg", "CondDropCfg", "CharCfg", "GlyphCfg", "CalligStyleCfg",
           "LocalCACfg", "DeformCfg", "DiTConfig"]


@dataclass
class CoreCfg:
    """主干尺寸 / 骨干当代 / 通道（CFG 作用域）"""
    input_size: int = 32
    num_calligraphers: Optional[int] = None
    num_characters: Optional[int] = None
    num_script_classes: int = 3
    num_scripts: int = 12
    learn_sigma: bool = False
    condition_fusion: str = "factorized_cat"
    cond_fusion_norm: str = "joint"
    in_channels: int = 4
    image_channels: int = 4
    norm_type: str = "rms"
    mlp_type: str = "swiglu"
    qk_norm: bool = True
    rope: bool = True
    rope_theta: float = 100.0
    attn_impl: str = "sdpa"


@dataclass
class CondDropCfg:
    """条件丢弃 / 门控 / 注入路由"""
    cond_drop_all_prob: Optional[float] = None
    cond_drop_one_prob: Optional[float] = None
    cond_drop_which_glyph_prob: float = 0.5
    cond_drop_callig_prob: float = 0.0
    cond_drop_script_prob: float = 0.0
    cond_drop_char_prob: float = 0.0
    cond_gate_t: float = 0.0
    cond_gate_floor: float = 0.0
    cond_inject_at: str = ""
    cond_inject_scale: bool = True
    xattn_q_pos: bool = False
    attn_tau: float = 1.0


@dataclass
class CharCfg:
    """汉字（内容）条件: 通路开关 + 三种字嵌入实现"""
    use_char_cond: bool = True
    content_cond: Optional[str] = None          # 'table' / 'std' / None(legacy 混合)
    char_embed_dim: Optional[int] = None
    char_proj_mode: str = "full"
    freeze_char_table: bool = False
    use_ids_char_embedder: bool = False
    ids_file: Optional[str] = None
    char_id_to_char: Optional[Dict[Any, Any]] = None
    use_std_dino_char_embedder: bool = False
    std_dino_table_path: Optional[str] = None


@dataclass
class GlyphCfg:
    """字形（标准字形软掩码 / 骨架）条件 —— std 模式的内容来源 ✓"""
    use_glyph_cond: bool = False
    std_soft_cond: bool = False
    std_soft_zero_init: bool = True
    glyph_scale_init: float = 0.4
    glyph_concat_input: bool = False
    glyph_gate_t: float = 0.0
    glyph_gate_floor: float = 0.35
    glyph_drop_prob: float = 0.0
    glyph_inject_layers: int = 0
    glyph_inject_mode: str = "adaln"
    glyph_inject_at: str = ""
    glyph_in_channels: int = 4
    glyph_embedder_depth: int = 0
    glyph_embedder_sep: bool = False
    glyph_vec_cond: bool = False
    glyph_vec_dim: int = 128
    glyph_vec_pool: str = "mean"


@dataclass
class CalligStyleCfg:
    """书家 / 书体 / 风格支路（含 S2 三层语义分解）"""
    callig_embed_dim: Optional[int] = None
    callig_proj_mode: str = "linear"
    callig_scale_init: float = 1.0
    callig_multi_style_k: int = 0
    callig_style_ca: bool = False
    callig_spatial: Any = False
    callig_spatial_rank: int = 64
    use_script_cond: bool = False
    script_embed_dim: int = 64
    style_token_n: int = 0
    style_role_init: float = 0.02
    style_ln: bool = False
    style_gain_init: float = 1.0
    style_y_over_t_init: float = 1.0
    style_ada_rank: int = 0
    # v85A: 只在指定层建风格支路（"8,10,11" / "8,10,11,final" / ""=全部层）
    style_ada_at: str = ""
    # v85A: Δmod 前的 tanh(style_gain) 门控（gain 初始 0 ⇒ warm-start 逐位等价）
    style_ada_gain: bool = False
    style_ctx_every_layer: bool = False
    hier_style: int = 0
    num_pairs: int = 0
    pair_init: str = "zero"
    pair_residual: int = 1
    script_film: bool = False
    spatial_film_rank: int = 0
    lowrank_spatial_rank: int = 0


@dataclass
class LocalCACfg:
    """局部风格-骨架引导（S2 local_ca）"""
    local_ca_layers: int = 0
    local_ca_heads: int = 4
    local_ca_rank: int = 64
    local_ca_q: str = "g"
    local_ca_window: int = 0
    local_ca_at: Optional[Any] = None
    local_ca_impl: str = "glyph_query"


@dataclass
class DeformCfg:
    """形变 / 残差 / 笔画调制头"""
    deform_skel: int = 0
    deform_width: int = 64
    deform_grid: int = 32
    deform_max_off: float = 3.0
    deform_coarse: int = 8
    deform_dt_ch: int = 0
    deform_topo: int = 0
    deform_warp_iters: int = 1
    deform_film_mode: str = "film"
    deform_style_tokens: int = 0
    deform_attn_heads: int = 4
    deform_preserve_amp: int = 0
    deform_ckpt: str = ""
    residual: int = 0
    res_cap: float = 1.0
    stroke_mod: int = 0
    stroke_cap: float = 1.0
    gate_radius: float = 0.25


@dataclass
class DiTConfig:
    """DiT 构造参数的唯一类型化载体（7 组 ✓）"""
    core: CoreCfg = field(default_factory=CoreCfg)
    cond: CondDropCfg = field(default_factory=CondDropCfg)
    char: CharCfg = field(default_factory=CharCfg)
    glyph: GlyphCfg = field(default_factory=GlyphCfg)
    style: CalligStyleCfg = field(default_factory=CalligStyleCfg)
    local_ca: LocalCACfg = field(default_factory=LocalCACfg)
    deform: DeformCfg = field(default_factory=DeformCfg)

    # ── Namespace/dict -> DiTConfig（**唯一映射** ✓，由 build.py 调用）──
    @classmethod
    def from_args(cls, a, overrides: Optional[Dict[str, Any]] = None) -> "DiTConfig":
        from src.model.build import derive                        # 复用派生值 ✓
        ov = overrides or {}
        g = lambda k, d=None: ov.get(k, getattr(a, k, d))         # noqa: E731
        gi = lambda k, dv=None: (None if g(k, dv) is None else int(g(k, dv)))      # noqa: E731
        gf = lambda k, dv=None: (None if g(k, dv) is None else float(g(k, dv)))    # noqa: E731
        d = derive(a, overrides)

        ids_map = None
        if bool(g("use_ids_char_embedder", False)):
            csv_path = g("ids_char_map_csv", None)
            if csv_path and os.path.isfile(csv_path):
                from src.model.ids_embedder import build_char_id_map_from_csv
                ids_map = build_char_id_map_from_csv(csv_path)

        return cls(
            core=CoreCfg(
                input_size=d["latent_size"],
                num_calligraphers=gi("num_calligraphers"),
                num_characters=gi("num_characters"),
                num_script_classes=gi("num_script_classes", 3) or 3,
                num_scripts=int(g("num_scripts", 12) or 12),
                learn_sigma=d["learn_sigma"],
                condition_fusion=g("condition_fusion"),
                cond_fusion_norm=g("cond_fusion_norm", "joint"),
                in_channels=d["in_channels"],
                image_channels=d["image_channels"],
                norm_type=g("norm_type", "rms"),
                mlp_type=g("mlp_type", "swiglu"),
                qk_norm=d["qk_norm"],
                rope=d["rope"],
                rope_theta=g("rope_theta", 100.0),
                attn_impl=g("attn_impl", "sdpa"),
            ),
            cond=CondDropCfg(
                cond_drop_all_prob=g("cond_drop_all_prob"),
                cond_drop_one_prob=g("cond_drop_one_prob"),
                cond_drop_which_glyph_prob=g("cond_drop_which_glyph_prob", 0.5),
                cond_drop_callig_prob=gf("cond_drop_callig_prob", 0.0) or 0.0,
                cond_drop_script_prob=gf("cond_drop_script_prob", 0.0) or 0.0,
                cond_drop_char_prob=gf("cond_drop_char_prob", 0.0) or 0.0,
                cond_gate_t=gf("cond_gate_t", 0.0) or 0.0,
                cond_gate_floor=gf("cond_gate_floor", 0.0) or 0.0,
                cond_inject_at=str(g("cond_inject_at", "") or ""),
                cond_inject_scale=bool(g("cond_inject_scale", True)),
                xattn_q_pos=bool(g("xattn_q_pos", False)),
                attn_tau=gf("attn_tau", 1.0) or 1.0,
            ),
            char=CharCfg(
                use_char_cond=d["use_char_cond"],
                content_cond=d["content_cond"],
                char_embed_dim=gi("char_embed_dim"),
                char_proj_mode=g("char_proj_mode", "full"),
                freeze_char_table=bool(g("freeze_char_table", False)),
                use_ids_char_embedder=bool(g("use_ids_char_embedder", False)),
                ids_file=g("ids_file", None),
                char_id_to_char=ids_map,
                use_std_dino_char_embedder=bool(g("use_std_dino_char_embedder", False)),
                std_dino_table_path=g("std_dino_table_path", None),
            ),
            glyph=GlyphCfg(
                use_glyph_cond=d["use_glyph_cond"],
                std_soft_cond=d["std_soft_cond"],
                std_soft_zero_init=bool(g("std_soft_zero_init", True)),
                glyph_scale_init=gf("glyph_scale_init", 0.4),
                glyph_concat_input=bool(g("glyph_concat_input", False)),
                glyph_gate_t=gf("glyph_gate_t", 0.0),
                glyph_gate_floor=gf("glyph_gate_floor", 0.35),
                glyph_drop_prob=gf("glyph_drop_prob", 0.0),
                glyph_inject_layers=gi("glyph_inject_layers", 0),
                glyph_inject_mode=g("glyph_inject_mode", "adaln"),
                glyph_inject_at=str(g("glyph_inject_at", "") or ""),
                glyph_in_channels=d["glyph_in_channels"],
                glyph_embedder_depth=gi("glyph_embedder_depth", 0),
                glyph_embedder_sep=bool(g("glyph_embedder_sep", False)),
                glyph_vec_cond=bool(g("glyph_vec_cond", False)),
                glyph_vec_dim=gi("glyph_vec_dim", 128),
                glyph_vec_pool=g("glyph_vec_pool", "mean"),
            ),
            style=CalligStyleCfg(
                callig_embed_dim=gi("callig_embed_dim"),
                callig_proj_mode=g("callig_proj_mode", "linear"),
                callig_scale_init=gf("callig_scale_init", 1.0),
                callig_multi_style_k=gi("callig_multi_style_k", 0),
                callig_style_ca=bool(g("callig_style_ca", False)),
                callig_spatial=g("callig_spatial", False),
                callig_spatial_rank=gi("callig_spatial_rank", 64),
                use_script_cond=bool(g("use_script_cond", False)),
                script_embed_dim=int(g("script_embed_dim", None) or 64),
                style_token_n=gi("style_token_n", 0),
                style_role_init=gf("style_role_init", 0.02),
                style_ln=bool(g("style_ln", False)),
                style_gain_init=gf("style_gain_init", 1.0),
                style_y_over_t_init=gf("style_y_over_t_init", 1.0),
                style_ada_rank=gi("style_ada_rank", 0),
            style_ada_at=g("style_ada_at", "") or "",
            style_ada_gain=bool(g("style_ada_gain", False)),
                style_ctx_every_layer=bool(g("style_ctx_every_layer", False)),
                hier_style=gi("hier_style", 0),
                num_pairs=gi("num_pairs", 0),
                pair_init=str(g("pair_init", "zero")),
                pair_residual=gi("pair_residual", 1),
                script_film=bool(g("script_film", False)),
                spatial_film_rank=gi("spatial_film_rank", 0),
                lowrank_spatial_rank=gi("lowrank_spatial_rank", 0),
            ),
            local_ca=LocalCACfg(
                local_ca_layers=gi("local_ca_layers", 0),
                local_ca_heads=gi("local_ca_heads", 4),
                local_ca_rank=gi("local_ca_rank", 64),
                local_ca_q=str(g("local_ca_q", "g")),
                local_ca_window=gi("local_ca_window", 0),
                local_ca_at=g("local_ca_at", None),
                local_ca_impl=str(g("local_ca_impl", "glyph_query")),
            ),
            deform=DeformCfg(
                deform_skel=gi("deform_skel", 0),
                deform_width=gi("deform_width", 64),
                deform_grid=gi("deform_grid", 32),
                deform_max_off=gf("deform_max_off", 3.0),
                deform_coarse=gi("deform_coarse", 8),
                deform_dt_ch=gi("deform_dt_ch", 0),
                deform_topo=gi("deform_topo", 0),
                deform_warp_iters=gi("deform_warp_iters", 1),
                deform_film_mode=str(g("deform_film_mode", "film")),
                deform_style_tokens=gi("deform_style_tokens", 0),
                deform_attn_heads=gi("deform_attn_heads", 4),
                deform_preserve_amp=gi("deform_preserve_amp", 0),
                deform_ckpt=str(g("deform_ckpt", "") or ""),
                residual=gi("residual", 0),
                res_cap=gf("res_cap", 1.0),
                stroke_mod=gi("stroke_mod", 0),
                stroke_cap=gf("stroke_cap", 1.0),
                gate_radius=gf("gate_radius", 0.25),
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        """展开成 `DiT_2Cond(**kwargs)` 需要的扁平 kwargs（字段名与旧参数逐字相同 ✓）"""
        out: Dict[str, Any] = {}
        for grp in fields(self):
            out.update(asdict(getattr(self, grp.name)))
        return out
