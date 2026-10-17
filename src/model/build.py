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

__all__ = ["aux_dirs_of", "derive", "resolve_content_cond", "dit_kwargs", "build_dit_from_args", "build_unet_from_args", "build_model_from_args"]

_DEFAULT_MODEL = "DiT-2Cond-Sp/2"


def aux_dirs_of(a) -> list:
    """辅助 latent 分片目录（与 src/train/train.py:98 同义 ✓）"""
    return [x for x in str(getattr(a, "aux_latent_shards_dirs", "") or "").split(",") if x]


def resolve_content_cond(a, log=None) -> Optional[str]:
    """内容条件模式: 'table' / 'std' / 'hybrid' / None(未显式声明 -> 推导并告警) ✓

    ★ 2026-10-13 [审查 PR2]: 语义**唯一真源**已收拢到 src/model/conditioning.py 的
      ContentConfig ✓ —— 本函数只是保持旧接口的薄封装（训练/评测都经由它 ✓）:
        table  : 字表给内容      (use_char_cond=True,  std_soft_cond=False)
        std    : 字形给内容      (use_char_cond=False, std_soft_cond=True)
        hybrid : 两者并存        (use_char_cond=True,  std_soft_cond=True)
                 ⇒ **归因不可分离** ✗（v83-A/B/C 的真实架构; 历史 ckpt 必须能如实标注 ✓）
      未显式写 content_cond 时按开关推导并告警 ✓（不再允许"隐式 hybrid"悄悄发生 ✗）。
    """
    from src.model.conditioning import content_config_from_args
    cfg = content_config_from_args(a, log=log)
    return (cfg.mode if cfg.declared else None)


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
    if ids_char_id_to_char is None and bool(g("use_ids_char_embedder", False)):
        csv_path = g("ids_char_map_csv", None)
        if csv_path and os.path.isfile(csv_path):
            from src.model.ids_embedder import build_char_id_map_from_csv
            ids_char_id_to_char = build_char_id_map_from_csv(csv_path)
    # ★ 2026-10-13: 手写 114 行 kwargs -> **类型化分组配置** src/model/config.py ✓
    #   新增参数只需加到对应 dataclass 组, 训练/评测自动同步（漏传->架构静默不一致 ✗ 这类 bug 结构性消失）
    from src.model.config import DiTConfig
    kw = DiTConfig.from_args(a, overrides=overrides).to_kwargs()
    if ids_char_id_to_char is not None:
        kw["char_id_to_char"] = ids_char_id_to_char   # 显式传入优先 ✓
    return kw


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
    mode = (overrides or {}).get('spatial_probe', getattr(a, 'spatial_probe', 'off'))
    if mode != 'off':
        from src.model.spatial_probe import attach_spatial_probe
        attach_spatial_probe(model, mode)
    if device is not None:
        model = model.to(device)
    model.eval() if eval_mode else None
    return model


def build_unet_from_args(a, device=None, *, overrides: Optional[Dict[str, Any]] = None, eval_mode: bool = False):
    """Flow Matching U-Net (FM-UNet) 统一构造入口。"""
    from src.model.fm_unet import FMUNet
    ov = overrides or {}
    g = lambda k, d=None: ov.get(k, getattr(a, k, d))
    model = FMUNet(
        in_channels=int(g("latent_channels", 4)),
        out_channels=int(g("latent_channels", 4)),
        base_channels=int(g("unet_base_channels", 128)),
        channel_mult=tuple(g("unet_channel_mult", (1, 2, 3))),
        cond_dim=int(g("cond_dim", 512)),
        num_calligraphers=int(g("num_calligraphers", 10)),
        callig_embed_dim=int(g("callig_embed_dim", 32)),
        num_script_classes=int(g("num_script_classes", 3)),
        script_embed_dim=int(g("script_embed_dim", 16)),
        time_freq_dim=int(g("time_freq_dim", 256)),
        num_res_blocks=int(g("unet_res_blocks", 2)),
        attn_resolutions=tuple(g("unet_attn_res", (16, 8))),
        num_groups=int(g("unet_num_groups", 32)),
    )
    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def build_model_from_args(a, device=None, *, ids_char_id_to_char=None,
                         overrides: Optional[Dict[str, Any]] = None, eval_mode: bool = False):
    """统一模型工厂：根据 a.model 自动分发 DiT 或 FM-UNet。"""
    name = str((overrides or {}).get("model", getattr(a, "model", None)) or _DEFAULT_MODEL)
    if name.startswith("FM-UNet") or name.startswith("UNet") or "unet" in name.lower():
        return build_unet_from_args(a, device=device, overrides=overrides, eval_mode=eval_mode)
    return build_dit_from_args(a, device=device, ids_char_id_to_char=ids_char_id_to_char,
                               overrides=overrides, eval_mode=eval_mode)

