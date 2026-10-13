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
from src.model.build import build_dit_from_args   # noqa: E402
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model_from_args(a, device, **overrides):
    """...

    expect_ckpt_weights: True 表示**随后会 load_state_dict 灌入已训权重** ⇒
      预训练表缺失/形状不匹配只告警不报错 ✓（2026-10-13 [审查 v3]: 恢复 ckpt 时
      不该要求外部表文件存在 ✗）。False = 从头初始化, 表是必需资源 ✓
    """
    """**严格复刻** train.py 的 ``DiT_2Cond_models[...]`` 构造（字段与默认值逐一对齐）。

    参数
    ----
    a : argparse.Namespace | dict
        ckpt 里的 ``args``。dict 也会被接受（自动转 Namespace）。
    device : str | torch.device
    overrides : 覆盖个别字段（调试用），例如 ``image_channels=4``

    返回已 ``.to(device).eval()`` 的模型（**未加载权重**）。
    """

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

    # ⚠ 2026-10-05: vae_downscale 的默认值必须与**训练侧**一致 (src/train/train.py 的训练侧默认值
    #   用的是 getattr(args, 'vae_downscale', 8))。原先这里写 4 -> 任何未显式提供该键的
    #   config 在评测/构造路径上会建出 2 倍分辨率的模型 (256 token -> 1024 token),
    #   pos_embed 形状失配; 在 --resume-full 后训练场景下更糟: 会被当"形状失配"剔除并
    #   重新随机初始化, 且因进了 _missing 而变成**可训** -> 静默坏掉底座。

    # ★ 2026-10-13: 构造逻辑收敛到**唯一实现** src/model/build.py（训练/评测共用）✓
    #   原先此处手写 ~110 行 kwargs 与 train.py 人肉同步, 是"评测架构静默不一致"的温床 ✗
    model = build_dit_from_args(a, device, overrides=overrides, eval_mode=True)
    return model
    return model.to(device).eval()


def _resolve(path):
    """相对路径按仓库根解析（脚本可能在别的 cwd 下运行）。"""
    if not path:
        return ""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(ROOT, path)
    return alt if os.path.exists(alt) else path


def apply_post_construction(model, a, verbose=True, expect_ckpt_weights=False):
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
        from src.model.dit import MultiStyleEmbedder
        if isinstance(model.y_callig_embedder, MultiStyleEmbedder):
            # v15: 表 (N, K*D) 不含 null 行, 全表覆盖; null_embed 独立参数保持随机
            if tuple(_emb.shape) == tuple(_w.shape):
                with torch.no_grad():
                    _w.copy_(_emb.float())
            elif verbose:
                # 加载已训 ckpt 时预训练表本就多余(随后 load_state_dict 会用 ckpt 训练后的
                # 表覆盖)。形状不匹配 -> 警告+跳过, 不再硬 assert (否则 v17 等 87-pair 模型
                # 因 ckpt 内嵌了旧的 45 表路径而无法 eval/探针)。
                print(f"[model_io] ⚠ 预训练多模态表 {tuple(_emb.shape)} != 模型表 "
                      f"{tuple(_w.shape)} -> 跳过灌表(以 ckpt 权重为准)。若这是**从头**训练"
                      f"请检查 num_calligraphers/callig_multi_style_k/callig_emb_pretrained 配套。")
        else:
            if tuple(_emb.shape) == (_w.shape[0] - 1, _w.shape[1]):
                with torch.no_grad():
                    _w[:_emb.shape[0]].copy_(_emb.float())
            elif verbose:
                print(f"[model_io] ⚠ 预训练书家表 {tuple(_emb.shape)} != 模型表 "
                      f"{tuple(_w.shape)} 去掉 null 行 -> 跳过灌表(以 ckpt 权重为准)。若这是"
                      f"**从头**训练请检查 num_calligraphers/callig_emb_pretrained 配套。")
        del _d
        if verbose:
            print(f"[model_io] callig 预训练表已加载: {tuple(_emb.shape)}")

    if bool(g("freeze_callig_table", False)):
        # ★ 2026-10-13 [审查 v3, 修正版]: 审查建议"恢复时缺外部表只告警" —— 但**不能只是跳过**
        #   ✗: freeze_table() 会把 CFG 的 null token 拆成独立 Parameter
        #   (y_callig_embedder.null_embed), **不调用这个键就不存在** ⇒ 恢复含该键的 ckpt 会
        #   Error(s) in loading state_dict ✗✗(本 session 实测确认 ✓)。
        #   正确语义: **始终执行冻结以保证键存在** ✓, 缺外部表时只**跳过灌入预训练值**
        #   （值随后由 load_state_dict 覆盖 ✓); 只有从头训练且无表时才当硬错误 ✗。
        if _cep and os.path.exists(_cep):
            model.y_callig_embedder.freeze_table()
            if verbose:
                print("[model_io] callig 表已冻结 (null_embed 已拆出)")
        elif expect_ckpt_weights:
            model.y_callig_embedder.freeze_table()
            if verbose:
                print("[model_io] ⚠ freeze_callig_table=True 但未找到 callig_emb_pretrained —— "
                      "仍执行冻结以生成 null_embed 键 ✓（预训练值缺失, 随后由 ckpt 覆盖 ✓）")
        else:
            raise ValueError(
                "freeze_callig_table=True 需要 callig_emb_pretrained（从头训练时否则 "
                "null_embed 无法确定）")
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
    # ★ 2026-10-13 [审查 P0]: 训练器保存**主权重**用的键历史上是 "delta", 而这里只读
    #   "model" ⇒ 无 EMA 的 ckpt 会 KeyError ✗。现在显式按 ema > model > delta 取,
    #   并**打印来源**, 不再靠猜 ✓
    _cands = (["ema", "model", "delta"] if _use_ema else ["model", "delta", "ema"])
    _src = next((k for k in _cands if ck.get(k) is not None), None)
    if _src is None:
        raise KeyError(f"ckpt 里找不到权重(尝试 {_cands}); 实际键={sorted(ck.keys())[:10]}")
    sd = ck[_src]
    print(f"[model_io] weights from '{_src}'" + (" (use_ema=True 但无 ema, 已回退 ✓)"
          if (_use_ema and _src != "ema") else ""))
    if verbose:
        print(f"[model_io] weights from {'ema' if _use_ema else 'model'}")

    # ★ 兼容旧 ckpt: local_ca 有两代实现，state_dict 键不同。**以 state_dict 为准**
    #   自动判定 —— 否则旧 ckpt 在 strict=True 下直接 RuntimeError 加载失败。
    #   旧: local_ca.N.{cond_norm, style_qk.*, res_logit, q/k/v/out_proj}
    #   新: local_ca.N.{style_to_q, style_to_k, out_log_scale, q/k/v/out_proj}
    _lc_keys = [k for k in sd.keys() if ".local_ca." in k]
    if _lc_keys:
        _legacy = any((".style_qk." in k or ".cond_norm." in k
                       or k.endswith(".res_logit")) for k in _lc_keys)
        try:
            a.local_ca_impl = "legacy" if _legacy else "glyph_query"
        except Exception:
            pass
        if verbose:
            print(f"[model_io] local_ca_impl = {getattr(a, 'local_ca_impl', '?')} "
                  f"(按 state_dict 键自动判定, {len(_lc_keys)} 个 local_ca 键)")

    model = build_model_from_args(a, device, expect_ckpt_weights=True,
                              **(model_overrides or {}))
    apply_post_construction(model, a, verbose=verbose,
                                 expect_ckpt_weights=True)

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
