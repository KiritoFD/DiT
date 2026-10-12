# -*- coding: utf-8 -*-
"""build.py — **模型构造的唯一实现**（训练与评测共用）

为什么要有这个文件
------------------
`src/train/train.py` 与 `src/eval/model_io.py` 原本各自手写一份 ~110–150 行的
`DiT_2Cond(...)` kwargs 清单，靠人肉同步。评测侧 docstring 还写着"**严格复刻**
train.py 的构造"，而它每一行旁边都留着事故记录：

    缺这行 -> 评测建出没有 X 的模型 -> X 的权重变 unexpected key
          -> strict=False 静默走旧通路 -> **指标全错**

这类"训练/评测架构不一致"是**静默错误**，指标照样算得出来。⇒ 收敛成一份 ✓

用法
----
    from src.model.build import build_dit_from_args, resolve_content_cond
    model = build_dit_from_args(args, device=dev)         # 训练
    model = build_dit_from_args(ckpt_args, overrides={...})  # 评测

语义以 **训练侧** 为唯一真源（`src/train/train.py` 的构造前计算，逐字复刻 ✓）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

__all__ = ["aux_dirs_of", "derive", "resolve_content_cond", "dit_kwargs", "build_dit_from_args"]

_DEFAULT_MODEL = "DiT-2Cond-Sp/2"


def aux_dirs_of(a) -> list:
    """辅助 latent 分片目录（与 src/train/train.py:98 同义 ✓）"""
    return [x for x in str(getattr(a, "aux_latent_shards_dirs", "") or "").split(",") if x]


def resolve_content_cond(a, log=None) -> Optional[str]:
    """内容条件模式: 'table' / 'std' / None(legacy 混合, 仅告警) —— 与 train.py 同一规则 ✓

    table: 字表给内容 (use_char_cond=True,  std_soft_cond=False)
    std  : 字形给内容 (use_char_cond=False, std_soft_cond=True)
    两者**不得共存** ✗
    """
    std_soft = bool(getattr(a, "std_soft_cond", False))
    use_char = not bool(getattr(a, "no_char_cond", False))
    cc = getattr(a, "content_cond", None)
    if cc is not None:
        cc = str(cc).lower()
        if cc not in ("table", "std"):
            raise ValueError(f"content_cond 只能是 'table' 或 'std'，收到 {cc!r}")
        if cc == "table" and std_soft:
            raise ValueError("content_cond='table' 但 std_soft_cond=True —— 两种内容条件不得共存 ✗")
        if cc == "std" and use_char:
            raise ValueError("content_cond='std' 但字表仍开 —— 请设 no_char_cond=True ✗")
        if cc == "std" and not std_soft:
            raise ValueError("content_cond='std' 需要 std_soft_cond=True")
    elif use_char and std_soft and log is not None:
        log("[content-cond] ⚠ legacy 混合模式: 字表与 std 字形同时开启 ✗ 归因不可分离")
    return cc


def derive(a, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造前派生值（逐字复刻 src/train/train.py:433-469 的语义 ✓）"""
    ov = overrides or {}
    g = lambda k, d=None: ov.get(k, getattr(a, k, d))          # noqa: E731
    dt = str(g("diffusion_type", "ddpm")).lower()
    ls = g("learn_sigma", None)
    n_aux = len(aux_dirs_of(a))
    lat_ch = int(g("latent_channels", 4))
    img_ch = g("image_channels", None)
    img_ch = lat_ch if img_ch is None else int(img_ch)
    if img_ch != lat_ch:
        raise AssertionError(
            f"image_channels({img_ch}) 必须等于 latent_channels({lat_ch}) —— 它是 **CFG 作用域**, "
            f"不能跟着 aux 一起放大, 否则 aux 通道被异分布的同一 cfg_scale 放大 -> 采样轨迹跑飞")
    std_soft = bool(g("std_soft_cond", False))
    return {
        "diffusion_type": dt,
        "learn_sigma": bool(ls) if ls is not None else dt not in ("flow", "flow_matching", "fm"),
        "qk_norm": bool(g("qk_norm", True)),
        "rope": bool(g("rope", True)),
        "n_aux": n_aux,
        "latent_channels": lat_ch,
        "image_channels": img_ch,
        "in_channels": lat_ch + 4 * n_aux,
        "glyph_in_channels": (1 if std_soft else int(g("glyph_latent_channels", 4) or 4)),
        "use_glyph_cond": (g("w_glyph_cond", 0) > 0
                           or bool(g("skel_as_glyph_cond", False)) or std_soft),
        "std_soft_cond": std_soft,
        "use_char_cond": not bool(g("no_char_cond", False)),
        "content_cond": resolve_content_cond(a),
        "latent_size": int(g("image_size", 256)) // int(g("vae_downscale", 8) or 8),
    }


def dit_kwargs(a, overrides: Optional[Dict[str, Any]] = None,
               ids_char_id_to_char=None) -> Dict[str, Any]:
    """产出 DiT_2Cond 的全部构造 kwargs（**唯一**映射 ✓）"""
    ov = overrides or {}
    g = lambda k, d=None: ov.get(k, getattr(a, k, d))          # noqa: E731
    d = derive(a, overrides)
    gi = lambda k, dv=None: (None if g(k, dv) is None else int(g(k, dv)))    # noqa: E731
    gf = lambda k, dv=None: (None if g(k, dv) is None else float(g(k, dv)))  # noqa: E731
    if ids_char_id_to_char is None and bool(g("use_ids_char_embedder", False)):
        csv_path = g("ids_char_map_csv", None)
        if csv_path and os.path.isfile(csv_path):
            from src.model.ids_embedder import build_char_id_map_from_csv
            ids_char_id_to_char = build_char_id_map_from_csv(csv_path)
    return dict(
        input_size=d["latent_size"],
        num_calligraphers=gi("num_calligraphers"),
        num_characters=gi("num_characters"),
        learn_sigma=d["learn_sigma"],
        condition_fusion=g("condition_fusion"),
        cond_fusion_norm=g("cond_fusion_norm", "joint"),
        callig_embed_dim=gi("callig_embed_dim"),
        char_embed_dim=gi("char_embed_dim"),
        glyph_vec_cond=bool(g("glyph_vec_cond", False)),
        glyph_vec_dim=gi("glyph_vec_dim", 128),
        glyph_vec_pool=g("glyph_vec_pool", "mean"),
        cond_drop_all_prob=g("cond_drop_all_prob"),
        cond_drop_one_prob=g("cond_drop_one_prob"),
        cond_drop_which_glyph_prob=g("cond_drop_which_glyph_prob", 0.5),
        cond_drop_callig_prob=float(g("cond_drop_callig_prob", 0.0) or 0.0),
        cond_drop_script_prob=float(g("cond_drop_script_prob", 0.0) or 0.0),
        cond_drop_char_prob=float(g("cond_drop_char_prob", 0.0) or 0.0),
        use_script_cond=bool(g("use_script_cond", False)),
        use_glyph_cond=d["use_glyph_cond"],
        std_soft_cond=d["std_soft_cond"],
        std_soft_zero_init=bool(g("std_soft_zero_init", True)),
        use_char_cond=d["use_char_cond"],
        content_cond=d["content_cond"],
        glyph_scale_init=gf("glyph_scale_init", 0.4),
        glyph_concat_input=bool(g("glyph_concat_input", False)),
        glyph_gate_t=gf("glyph_gate_t", 0.0),
        glyph_gate_floor=gf("glyph_gate_floor", 0.35),
        cond_gate_t=float(g("cond_gate_t", 0.0) or 0.0),
        cond_gate_floor=float(g("cond_gate_floor", 0.0) or 0.0),
        glyph_drop_prob=gf("glyph_drop_prob", 0.0),
        glyph_inject_layers=gi("glyph_inject_layers", 0),
        glyph_inject_mode=g("glyph_inject_mode", "adaln"),
        glyph_inject_at=str(g("glyph_inject_at", "") or ""),
        cond_inject_at=str(g("cond_inject_at", "") or ""),
        cond_inject_scale=bool(g("cond_inject_scale", True)),
        xattn_q_pos=bool(g("xattn_q_pos", False)),
        attn_tau=float(g("attn_tau", 1.0) or 1.0),
        glyph_embedder_depth=gi("glyph_embedder_depth", 0),
        glyph_embedder_sep=bool(g("glyph_embedder_sep", False)),
        style_token_n=gi("style_token_n", 0),
        style_role_init=gf("style_role_init", 0.02),
        glyph_in_channels=d["glyph_in_channels"],
        in_channels=d["in_channels"],
        char_proj_mode=g("char_proj_mode", "full"),
        callig_proj_mode=g("callig_proj_mode", "linear"),
        callig_scale_init=gf("callig_scale_init", 1.0),
        style_ln=bool(g("style_ln", False)),
        style_gain_init=gf("style_gain_init", 1.0),
        style_y_over_t_init=gf("style_y_over_t_init", 1.0),
        style_ada_rank=gi("style_ada_rank", 0),
        callig_multi_style_k=gi("callig_multi_style_k", 0),
        callig_style_ca=bool(g("callig_style_ca", False)),
        style_ctx_every_layer=bool(g("style_ctx_every_layer", False)),
        callig_spatial=g("callig_spatial", False),
        callig_spatial_rank=gi("callig_spatial_rank", 64),
        hier_style=gi("hier_style", 0),
        num_pairs=gi("num_pairs", 0),
        num_scripts=int(g("num_scripts", 12) or 12),
        script_embed_dim=int(g("script_embed_dim", None) or 64),
        pair_init=str(g("pair_init", "zero")),
        pair_residual=gi("pair_residual", 1),
        script_film=bool(g("script_film", False)),
        spatial_film_rank=gi("spatial_film_rank", 0),
        lowrank_spatial_rank=gi("lowrank_spatial_rank", 0),
        local_ca_layers=gi("local_ca_layers", 0),
        local_ca_heads=gi("local_ca_heads", 4),
        local_ca_rank=gi("local_ca_rank", 64),
        local_ca_q=str(g("local_ca_q", "g")),
        local_ca_window=gi("local_ca_window", 0),
        local_ca_at=g("local_ca_at", None),
        local_ca_impl=str(g("local_ca_impl", "glyph_query")),
        deform_skel=gi("deform_skel", 0),
        deform_width=gi("deform_width", 64),
        deform_grid=gi("deform_grid", 32),
        deform_max_off=gf("deform_max_off", 3.0),
        deform_coarse=gi("deform_coarse", 8),
        residual=gi("residual", 0),
        res_cap=gf("res_cap", 1.0),
        stroke_mod=gi("stroke_mod", 0),
        stroke_cap=gf("stroke_cap", 1.0),
        gate_radius=gf("gate_radius", 0.25),
        deform_dt_ch=gi("deform_dt_ch", 0),
        deform_topo=gi("deform_topo", 0),
        deform_warp_iters=gi("deform_warp_iters", 1),
        deform_film_mode=str(g("deform_film_mode", "film")),
        deform_style_tokens=gi("deform_style_tokens", 0),
        deform_attn_heads=gi("deform_attn_heads", 4),
        deform_preserve_amp=gi("deform_preserve_amp", 0),
        deform_ckpt=str(g("deform_ckpt", "") or ""),
        freeze_char_table=bool(g("freeze_char_table", False)),
        use_ids_char_embedder=bool(g("use_ids_char_embedder", False)),
        ids_file=g("ids_file", None),
        char_id_to_char=ids_char_id_to_char,
        use_std_dino_char_embedder=bool(g("use_std_dino_char_embedder", False)),
        std_dino_table_path=g("std_dino_table_path", None),
        norm_type=g("norm_type", "rms"),
        mlp_type=g("mlp_type", "swiglu"),
        qk_norm=d["qk_norm"],
        rope=d["rope"],
        rope_theta=g("rope_theta", 100.0),
        attn_impl=g("attn_impl", "sdpa"),
        image_channels=d["image_channels"],
    )


def build_dit_from_args(a, device=None, *, ids_char_id_to_char=None,
                        overrides: Optional[Dict[str, Any]] = None, eval_mode: bool = False):
    """**唯一**的 DiT 构造入口（训练/评测共用 ✓）。

    eval_mode=True 时返回 `.to(device).eval()`（评测口径）; 否则仅 `.to(device)`（训练口径）。
    """
    from src.model import DiT_2Cond_models
    kwargs = dit_kwargs(a, overrides=overrides, ids_char_id_to_char=ids_char_id_to_char)
    name = str(getattr(a, "model", None) or _DEFAULT_MODEL)
    if name not in DiT_2Cond_models:
        raise KeyError(f"未知模型 {name!r}; 可用: {list(DiT_2Cond_models.keys())}")
    model = DiT_2Cond_models[name](**kwargs)
    if device is not None:
        model = model.to(device)
    model.eval() if eval_mode else None
    return model
