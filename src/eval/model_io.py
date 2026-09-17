# -*- coding: utf-8 -*-
"""模型重建 / ckpt 加载的**唯一入口**。

## 为什么必须有这个模块

从 ckpt 的 `args` 重建 DiT 是一件**极易静默出错**的事：模型有 40+ 个构造参数，
漏一个就形状不匹配。而历史上仓库里有 **9 处独立实现**，其中 8 处用
`load_state_dict(strict=False)` —— 后果是：

  **形状不匹配的层被静默跳过，保持随机初始化，指标照样算得出来，只是全是错的。**

我们已经真实踩过两次：
  - `tools/eval_diversity.py` 的 `build_model` 漏了 `glyph_vec_cond`，
    v12 的 `cond_fusion.0.weight` ckpt 是 `[256]`（callig128+glyph_vec128）
    而重建给 `[128]` -> 静默加载随机 `cond_fusion`
  - `torch.compile` 的 `_orig_mod.` 前缀 / `freeze_table()` 拆出的 `null_embed`
    都会让 key 对不上

## 用法

    from src.eval.model_io import load_model_from_ckpt
    model, a = load_model_from_ckpt("path/to/ckpt.pt", device="cuda")

它内部做了 `strict=True` 的护栏 —— **字段漏一个就抛错，不会静默**。
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model_from_args(a, device, **overrides):
    """**严格复刻** train.py 的 ``DiT_2Cond_models[...]`` 构造（字段与默认值逐一对齐）。

    参数
    ----
    a : argparse.Namespace | dict
        ckpt 里的 ``args``。dict 也会被接受（自动转 Namespace）。
    device : str | torch.device
    overrides : 覆盖个别字段（调试用），例如 ``image_channels=4``

    返回已 ``.to(device).eval()`` 的模型（**未加载权重**）。
    """
    from src.model import DiT_2Cond_models

    if isinstance(a, dict):
        from argparse import Namespace
        a = Namespace(**a)

    def g(k, d=None):
        return overrides.get(k, getattr(a, k, d))

    def gi(k, d=None):
        """int 转换，但 **None 原样透传** —— 模型的 __init__ 自带该字段的默认值，
        强转 int(None) 会 TypeError。ckpt 的 args 里确实有若干字段是 None
        （实测 v12 的 char_embed_dim 就是 None）。"""
        v = g(k, d)
        return None if v is None else int(v)

    def gf(k, d=None):
        v = g(k, d)
        return None if v is None else float(v)

    latent_size = gi("image_size", 256) // gi("vae_downscale", 4)
    _n_aux = len([s for s in str(g("aux_latent_shards_dirs", "") or "").split(",") if s])
    _lc = gi("latent_channels", 4) or 4

    model = DiT_2Cond_models[g("model")](
        input_size=latent_size,
        num_calligraphers=gi("num_calligraphers"),
        num_characters=gi("num_characters"),
        use_checkpoint=bool(g("use_checkpoint", False)),
        learn_sigma=bool(g("learn_sigma", False)),
        condition_fusion=g("condition_fusion"),
        callig_embed_dim=gi("callig_embed_dim"),
        char_embed_dim=gi("char_embed_dim"),
        glyph_vec_cond=bool(g("glyph_vec_cond", False)),
        glyph_vec_dim=gi("glyph_vec_dim", 128),
        glyph_vec_pool=g("glyph_vec_pool", "mean"),
        cond_drop_all_prob=g("cond_drop_all_prob"),
        cond_drop_one_prob=g("cond_drop_one_prob"),
        cond_drop_which_glyph_prob=g("cond_drop_which_glyph_prob", 0.5),
        use_glyph_cond=bool((g("w_glyph_cond", 0) or 0) > 0
                            or g("skel_as_glyph_cond", False)),
        use_char_cond=not bool(g("no_char_cond", False)),
        glyph_scale_init=gf("glyph_scale_init", 0.4),
        glyph_drop_prob=gf("glyph_drop_prob", 0.0),
        glyph_inject_layers=gi("glyph_inject_layers", 0),
        glyph_inject_mode=g("glyph_inject_mode", "adaln"),
        glyph_embedder_depth=gi("glyph_embedder_depth", 0),
        glyph_embedder_sep=bool(g("glyph_embedder_sep", False)),
        style_token_n=gi("style_token_n", 0),
        style_role_init=gf("style_role_init", 0.02),
        glyph_in_channels=4,
        in_channels=(_lc + 4 * _n_aux),
        char_proj_mode=g("char_proj_mode", "full"),
        callig_proj_mode=g("callig_proj_mode", "linear"),
        callig_scale_init=gf("callig_scale_init", 1.0),
        callig_style_attn=bool(g("callig_style_attn", False)),
        callig_n_style=gi("callig_n_style", 8),
        callig_spatial=bool(g("callig_spatial", False)),
        callig_spatial_rank=gi("callig_spatial_rank", 64),
        freeze_char_table=bool(g("freeze_char_table", False)),
        use_ids_char_embedder=bool(g("use_ids_char_embedder", False)),
        ids_file=g("ids_file", None),
        char_id_to_char=None,
        use_std_dino_char_embedder=bool(g("use_std_dino_char_embedder", False)),
        std_dino_table_path=g("std_dino_table_path", None),
        norm_type=g("norm_type", "rms"),
        mlp_type=g("mlp_type", "swiglu"),
        qk_norm=bool(g("qk_norm", True)),
        rope=bool(g("rope", True)),
        rope_theta=gf("rope_theta", 100.0),
        attn_impl=g("attn_impl", "sdpa"),
        image_channels=(g("image_channels")
                        if g("image_channels", None) is not None else _lc),
    )
    return model.to(device).eval()


def _resolve(path):
    """相对路径按仓库根解析（脚本可能在别的 cwd 下运行）。"""
    if not path:
        return ""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(ROOT, path)
    return alt if os.path.exists(alt) else path


def apply_post_construction(model, a, verbose=True):
    """复刻 train.py 在 ``DiT_2Cond_models[...]`` **之后**的两步后处理。

    不做这两步，``state_dict`` 的 key 会对不上（`strict=True` 下会抛错）：
      1. 预训练书家表：覆盖 ``[0, N)`` 行，null 行保持随机
      2. ``freeze_callig_table``：会把 CFG 的 null token **拆成独立 Parameter**
         ``y_callig_embedder.null_embed`` —— 不调用就没有这个 key
    """
    def g(k, d=None):
        return getattr(a, k, d)

    _cep = _resolve(str(g("callig_emb_pretrained", "") or ""))
    if _cep and os.path.exists(_cep):
        _d = torch.load(_cep, map_location="cpu", weights_only=False)
        _emb = _d["embedding"] if isinstance(_d, dict) else _d
        _w = model.y_callig_embedder.embedding_table.weight
        assert _emb.shape == (_w.shape[0] - 1, _w.shape[1]), (
            f"预训练书家表形状 {tuple(_emb.shape)} != 模型表 {tuple(_w.shape)} 去掉 null 行。"
            f" 检查 num_calligraphers / callig_emb_pretrained 是否配套")
        with torch.no_grad():
            _w[:_emb.shape[0]].copy_(_emb.float())
        del _d
        if verbose:
            print(f"[model_io] callig 预训练表已加载: {tuple(_emb.shape)}")

    if bool(g("freeze_callig_table", False)):
        assert _cep and os.path.exists(_cep), (
            "freeze_callig_table=True 需要 callig_emb_pretrained（否则 null_embed 无法确定）")
        model.y_callig_embedder.freeze_table()
        if verbose:
            print("[model_io] callig 表已冻结 (null_embed 已拆出)")

    # 双轴 CFG 开关（与 train.py 一致，挂模型属性，供 inference.sample_latents 透传）
    model.cfg_glyph_scale = (None if g("cfg_glyph_scale", None) is None
                             else float(g("cfg_glyph_scale")))
    model.cfg_w_inter = float(g("cfg_w_inter", 0.0) or 0.0)
    if model.cfg_glyph_scale is not None and float(g("glyph_drop_prob", 0.0) or 0.0) <= 0.0:
        print("[model_io] ⚠ 设了 cfg_glyph_scale 但 glyph_drop_prob=0 -> "
              "内容轴(g=0)在训练中从未出现过，双轴 CFG 的 content 轴无效")
    return model


def check_state_dict(model, sd, tag="", allow_missing=(), allow_unexpected=(),
                     raise_on_missing=True):
    """``strict=False`` 加载后的**显式检查** —— 把 missing 从静默变成报错。

    仓库里还有 ~80 处 ``strict=False``，多数是**合法的**部分加载
    （resume / 只灌 backbone）。但"**从 ckpt args 重建模型 + strict=False**"
    这一类是危险的：形状不匹配的层会保持随机初始化，指标照样算得出来。

    这个助手给那些站点一条迁移路径：保留 ``strict=False`` 的灵活性，
    但**显式列出**哪些键被允许缺失，其余一律抛错。

    用法::

        miss, unexp = model.load_state_dict(sd, strict=False)
        check_state_dict(model, sd, tag="batch_eval", allow_missing=("callig_style.*",))

    allow_missing / allow_unexpected : 允许的键前缀（fnmatch 风格子串匹配）
    """
    import fnmatch
    miss, unexp = [], []
    cur = model.state_dict()
    for k, v in sd.items():
        if k in cur:
            if tuple(cur[k].shape) != tuple(v.shape):
                miss.append(f"{k} (shape {tuple(v.shape)} -> {tuple(cur[k].shape)})")
        elif not any(fnmatch.fnmatch(k, p) or p in k for p in allow_unexpected):
            unexp.append(k)
    for k in cur:
        if k not in sd and not any(fnmatch.fnmatch(k, p) or p in k for p in allow_missing):
            miss.append(k)
    if unexp:
        print(f"[model_io] ⚠ {tag}: unexpected 键 {len(unexp)}: {sorted(unexp)[:6]}")
    if miss:
        msg = (f"[model_io] ✗ {tag}: {len(miss)} 个键**形状不匹配或缺失** "
               f"-> 这些层会保持随机初始化，指标不可信！\n"
               f"    {sorted(miss)[:8]}\n"
               f"    修法: 用 src/eval/model_io.build_model_from_args 重建，"
               f"或把确实允许缺失的键加进 allow_missing")
        if raise_on_missing:
            raise RuntimeError(msg)
        print(msg)
    return miss, unexp


def load_model_from_ckpt(ckpt_path, device="cuda", use_ema=True, verbose=True,
                         model_overrides=None):
    """加载 ckpt -> ``(model, args)``。**strict=True 当护栏。**

    use_ema : True 时优先用 ``ck["ema"]``（推理口径），缺失则回退 ``ck["model"]``
    返回的 model 已 ``eval()``，并已挂好 ``cfg_glyph_scale`` / ``cfg_w_inter``。
    """
    if verbose:
        print(f"[model_io] loading {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if isinstance(a, dict):
        from argparse import Namespace
        a = Namespace(**a)
    _use_ema = bool(use_ema) and ("ema" in ck) and (ck["ema"] is not None)
    sd = ck["ema"] if _use_ema else ck["model"]
    if verbose:
        print(f"[model_io] weights from {'ema' if _use_ema else 'model'}")

    model = build_model_from_args(a, device, **(model_overrides or {}))
    apply_post_construction(model, a, verbose=verbose)

    # torch.compile 会加 `_orig_mod.` 前缀 -> 剥掉，否则 strict 直接报 key 不匹配
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        if verbose:
            print("[model_io] 已剥离 torch.compile 的 `_orig_mod.` 前缀")

    model.load_state_dict(sd, strict=True)
    del ck, sd
    model.eval()
    if verbose:
        print(f"[model_io] loaded strict=True OK "
              f"(params={sum(p.numel() for p in model.parameters()):,})")
    return model, a
