# -*- coding: utf-8 -*-
"""dit_conditioning.py — `DiT_2Cond` 的条件构造/融合/前向（2026-10-13 [审查 PR3] 结构归位 ✓）

审查 PR3 目标结构要求把条件相关逻辑从模型类里分出来 ✓。本模块承载**已抽出并验证过**的
8 个配置 dataclass + 8 个纯函数（合计约 1210 行 ✓）：

    CondDropCfg / cond_drop_masks                  条件 dropout 四路遮罩
    FuseCfg / fuse_condition                       条件融合
    CfgForwardCfg / forward_with_cfg               CFG 前向
    TwoAxisCfg / forward_with_2axis_cfg            双轴 CFG 前向
    StyleBranchCfg / style_branch                  风格分支
    EmbedGlyphCfg / embed_glyph                    字形条件嵌入
    InitWeightsCfg / initialize_weights            权重初始化
    ForwardCfg / forward                           主前向（组合层）

⚠ 关键约束（审查原文 ✓）: **不移动任何权重注册位置** ✓ —— 本模块**只含函数与冻结
  dataclass**，不定义任何 `nn.Module` 子类、不做任何 `self.X = nn.…` 注册 ✓
  ⇒ `DiT_2Cond` 的 **state_dict 键完全不变** ✓✓（由 strict 加载 + 三向前向黄金裁决 ✓）。

⚠ 再导出: `src/model/dit.py` 会 `from .dit_conditioning import (...)` 再导出这些名字 ✓
  ⇒ `from src.model.dit import forward / FuseCfg / ...` 的既有引用**全部不变** ✓。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .components import get_2d_sincos_pos_embed

__all__ = [
    'CondDropCfg',
    'cond_drop_masks',
    'FuseCfg',
    'fuse_condition',
    'CfgForwardCfg',
    'forward_with_cfg',
    'TwoAxisCfg',
    'forward_with_2axis_cfg',
    'StyleBranchCfg',
    'style_branch',
    'EmbedGlyphCfg',
    'embed_glyph',
    'InitWeightsCfg',
    'initialize_weights',
    'ForwardCfg',
    'forward',
]


@dataclass(frozen=True)
class CondDropCfg:
    """条件 dropout 的**全部**配置（2026-10-13 [审查 PR3] 从 `DiT_2Cond` 抽出 ✓）。

    审查: "第一阶段只移动逻辑, 不移动权重注册位置" ✓ —— 本类只把 `_cond_drop_masks`
    用到的 12 个 `self.*` **显式化** ✓，不注册任何模块 ⇒ **state_dict 键完全不变** ✓✓。
    """
    training: bool
    glyph_drop_prob: float
    use_char_cond: bool
    use_script_cond: bool
    condition_fusion: str
    cond_drop_all_prob: float
    cond_drop_one_prob: float
    cond_drop_callig_prob: float
    cond_drop_char_prob: float
    cond_drop_script_prob: float
    cond_drop_which_glyph_prob: float
    num_callig_classes: int

    @classmethod
    def from_model(cls, m):
        def _g(k, d):
            return getattr(m, k, d)
        return cls(training=bool(m.training),
                   glyph_drop_prob=float(_g("glyph_drop_prob", 0.0)),
                   use_char_cond=bool(_g("use_char_cond", True)),
                   use_script_cond=bool(_g("use_script_cond", False)),
                   condition_fusion=str(_g("condition_fusion", "")),
                   cond_drop_all_prob=float(_g("cond_drop_all_prob", 0.0)),
                   cond_drop_one_prob=float(_g("cond_drop_one_prob", 0.0)),
                   cond_drop_callig_prob=float(_g("cond_drop_callig_prob", 0.0)),
                   cond_drop_char_prob=float(_g("cond_drop_char_prob", 0.0)),
                   cond_drop_script_prob=float(_g("cond_drop_script_prob", 0.0)),
                   cond_drop_which_glyph_prob=float(_g("cond_drop_which_glyph_prob", 0.5)),
                   num_callig_classes=int(m.y_callig_embedder.num_classes))


def cond_drop_masks(cfg, g, y_callig):
    """字形/书家/字/书体 四路条件丢弃遮罩（**train-only** ✓）。

    从 `DiT_2Cond._cond_drop_masks` 原样搬出（2026-10-13 [审查 PR3] 第一步 ✓），
    只把 `self.*` 换成 `cfg.*` ✓；`torch.rand` 的**调用顺序完全不变** ✓
    ⇒ 固定种子下与搬移前**逐位等价** ✓（由 eval + train 两个前向黄金裁决 ✓）。

    返回 (g, keep, callig_drop, char_drop, script_drop, y_callig_in)。

    ⚠ **诚实记录（含一条我自己的教训）**: 迁移过程中我曾一度断言这里有一个"属性名少
      `_prob`"的潜伏 bug ✗ —— **那是错的** ✓。原因是我用 `python -c` 打印源码时用了
      `[:108]` **截断**，看到 `self.cond_drop_which_g` 就下了结论 ✗✗；用 read 工具读**完整**
      行后确认原文写的就是 `self.cond_drop_which_glyph_prob` ✓（名字正确 ✓）。
      ⇒ 教训: **报 bug 前必须读完整行，绝不从截断输出下诊断** ✓✓。
      本函数因此**不含任何行为修复** ✓：只是把 `self.*` 原样翻译成 `cfg.*` ✓。
    """
    if (g is not None and cfg.training and cfg.glyph_drop_prob > 0.0):
        N = g.shape[0]
        keep = torch.rand(N, device=g.device) >= cfg.glyph_drop_prob
        g = g * keep.view(N, 1, 1, 1).to(g.dtype)
    else:
        keep = None

    # ── callig 条件 drop mask: 在 g 路径**之前**计算并全程共享 ────────────
    # (adaLN 分支与骨架风格注入必须用同一份 drop mask, 否则 drop-callig 样本
    #  的 adaLN 是 null 向量而骨架通路仍带真实风格 -> uncond 分支被污染)
    callig_drop = None
    char_drop = None
    script_drop = None          # 书体条件 drop mask（仅 use_script_cond 且训练时才有值 ✓）
    # ⚠ 这个元组必须与**所有会读 `y_callig_in` / `char_drop` 的融合分支**一一对应。
    #   2026-09-17 修: 原先漏了 "factorized_cat" —— 该分支在下方同样复用
    #   `y_callig_in` / `char_drop`, 但因不在元组里, 整个 drop 块被跳过 ->
    #   `callig_drop is None` -> `y_callig_in = y_callig` -> **训练全程零条件 dropout**,
    #   null token 从未出现, `null_embed` 停在随机初始化。
    #   后果: 推理时 `forward_with_cfg` 的 uncond 半用的是**未训练**的 null 向量,
    #   `cfg=0.7` 实际是在往一个随机方向插值, 而不是真正的 CFG。
    #   不报错、loss 正常下降 —— 又一例静默失效。**新增 fusion 分支时必须回来加。**
    if cfg.use_script_cond and cfg.training:
        # v53 三表: 各因子**独立** dropout, 三者全中 = uncond;
        # cond_drop_all_prob 仍作无条件底座 (保 CFG 有足量 uncond 样本)。
        _B = y_callig.shape[0]
        _dev = y_callig.device
        _all = torch.rand(_B, device=_dev) < cfg.cond_drop_all_prob
        callig_drop = (torch.rand(_B, device=_dev) < cfg.cond_drop_callig_prob) | _all
        script_drop = (torch.rand(_B, device=_dev) < cfg.cond_drop_script_prob) | _all
        char_drop = (torch.rand(_B, device=_dev) < cfg.cond_drop_char_prob) | _all
    elif (cfg.condition_fusion in ("factorized_add", "factorized_cat", "xl_highdim")
            and cfg.training
            and (cfg.cond_drop_all_prob > 0 or cfg.cond_drop_one_prob > 0)):
        r = torch.rand(y_callig.shape[0], device=y_callig.device)
        if not cfg.use_char_cond:
            # v10b 单向量因子: drop_all/drop_one 同义
            callig_drop = r < (cfg.cond_drop_all_prob + cfg.cond_drop_one_prob)
        else:
            drop_all = r < cfg.cond_drop_all_prob
            drop_one = ((r >= cfg.cond_drop_all_prob)
                        & (r < cfg.cond_drop_all_prob + cfg.cond_drop_one_prob))
            which_glyph = torch.rand(y_callig.shape[0],
                                     device=y_callig.device) < cfg.cond_drop_which_glyph_prob
            callig_drop = drop_all | (drop_one & which_glyph)
            char_drop = drop_all | (drop_one & ~which_glyph)
    y_callig_in = (torch.where(callig_drop, cfg.num_callig_classes, y_callig)
                   if callig_drop is not None else y_callig)
    return g, keep, callig_drop, char_drop, script_drop, y_callig_in


@dataclass(frozen=True)
class FuseCfg:
    """`_fuse_condition` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 显式化 ✓）。

    审查: "第一阶段只移动逻辑, 不移动权重注册位置" ✓ 且 "通过**明确的数据对象**交换结果" ✓
    ⇒ 本类不注册任何模块 ⇒ **state_dict 键完全不变** ✓✓（strict 加载已复验 ✓）。
    ⚠ 其中两个字段是**绑定方法**（`embed_callig` / `style_branch` ✓）—— 原为 self._embed_callig /
      self._style_branch ✓；其余是 nn.Module 或配置标量 ✓。
    """
    embed_callig: Any                    # 原 self._embed_callig（绑定方法 ✓）
    style_branch: Any                    # 原 self._style_branch（绑定方法 ✓）
    callig_proj: Any
    callig_scale: Any
    char_proj: Any
    char_scale: Any
    cond_fusion: Any
    cond_gate_floor: Any
    cond_gate_t: Any
    cond_ln: Any
    condition_fusion: Any
    glyph_vec_cond: Any
    glyph_vec_out: Any
    glyph_vec_scale: Any
    training: Any
    use_char_cond: Any
    use_script_cond: Any
    y_callig_embedder: Any
    y_char_embedder: Any
    y_scale: Any
    y_script_embedder: Any

    @classmethod
    def from_model(cls, m):
        """★ 全部用 `getattr` 兜底 ✓✓ —— 原实现里这些属性是**分支内惰性访问** ✓：
        例如 `y_scale` 只在 use_char_cond 的某条路径上被读 ✓；若在此**急切**读取 ✗，
        则对不含该属性的模型（实测 v83-A ✗）会 `AttributeError` ✓。
        ⇒ 兜底为 None 后, 行为与原实现**完全一致** ✓（分支不走到就不会被读 ✓），
          由 eval + train 两个前向黄金**逐位**裁决 ✓✓。
        """
        def _g(k):
            return getattr(m, k, None)
        return cls(
                   embed_callig=m._embed_callig,
                   style_branch=m._style_branch,
                   callig_proj=_g("callig_proj"),
                   callig_scale=_g("callig_scale"),
                   char_proj=_g("char_proj"),
                   char_scale=_g("char_scale"),
                   cond_fusion=_g("cond_fusion"),
                   cond_gate_floor=_g("cond_gate_floor"),
                   cond_gate_t=getattr(m, "cond_gate_t", 0.0),   # ★ 默认必须与原 getattr 一致 ✓
                   cond_ln=_g("cond_ln"),
                   condition_fusion=_g("condition_fusion"),
                   glyph_vec_cond=_g("glyph_vec_cond"),
                   glyph_vec_out=_g("glyph_vec_out"),
                   glyph_vec_scale=_g("glyph_vec_scale"),
                   training=m.training,
                   use_char_cond=_g("use_char_cond"),
                   use_script_cond=_g("use_script_cond"),
                   y_callig_embedder=_g("y_callig_embedder"),
                   y_char_embedder=_g("y_char_embedder"),
                   y_scale=_g("y_scale"),
                   y_script_embedder=_g("y_script_embedder"),
        )


def fuse_condition(cfg, t, x, y_callig_in, y_callig, y_char, y_script,
                   char_drop, script_drop, e_glyph_vec, g_tok, e_style, style_tokens):
    """条件融合（**逐行等价**于 `DiT_2Cond._fuse_condition` ✓，2026-10-13 [审查 PR3] 第二步 ✓）。

    只把 `self.X` 换成 `cfg.X` ✓；**未改动任何数值逻辑、未新增/移动任何 nn.Module** ✓
    ⇒ 由 eval + train 两个前向黄金**逐位**裁决 ✓✓。
    """
    """条件向量融合 -> y_emb（原 forward 内联 115 行 ✓）。

    四种 condition_fusion 分支 + Time-Gate；返回 (y_emb, e_callig, _cgw)，
    其中 e_callig 供风格专用支路 c_style 使用。
    """
    if cfg.condition_fusion == "factorized_add":
        # V3-A 二因子可组合 mask（4-way）：
        #   - drop_all           -> unconditional（CFG 基准）
        #   - drop_one & which=0 -> glyph-only（drop callig，学 content score s_G）
        #   - drop_one & which=1 -> callig-only（drop glyph，学 style score s_A）
        #   - 其余               -> full（callig+glyph，学 joint score）
        # 默认 0.10/0.30 配比 => full 60% / callig-only 15% / glyph-only 15% / uncond 10%。
        # 书家维度样本充足而字符维度才是难点: cond_drop_which_glyph_prob 让 drop-one
        # 偏向 glyph-only（drop callig 保 char），把专门训练预算给 5461 个字符内容分。
        # 注意: drop mask 已在 forward 顶部计算 (y_callig_in 与骨架风格注入共享),
        # 这里不再重复 roll (重复 roll 会导致两分支 mask 不一致)。
        if not cfg.use_char_cond:
            # v10b: 单向量因子 (callig), drop_all/drop_one 同义 —— 丢 callig = uncond 向量
            e_callig = cfg.embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens)
            y_emb = cfg.callig_scale * cfg.callig_proj(e_callig)
            y_emb = cfg.style_branch(e_callig, y_emb)
        else:
            if cfg.training and char_drop is not None:
                y_char = torch.where(char_drop, cfg.y_char_embedder.num_classes, y_char)
            e_callig = cfg.embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens)
            e_char = cfg.y_char_embedder(y_char, False)
            # 可学习幅度平衡：见 __init__ 处注释（DINO 区分度被书家分支淹没的实测）。
            y_emb = (cfg.callig_scale * cfg.callig_proj(e_callig)
                     + cfg.char_scale * cfg.char_proj(e_char)) / math.sqrt(2.0)
            y_emb = cfg.style_branch(e_callig, y_emb)
        if cfg.glyph_vec_cond and e_glyph_vec is not None:
            # g 向量因子: 与 callig 分支对称的"独立投影 + 可学习标量"加数。
            # 操作数集合与 factorized_cat 完全一致 -> 两种融合方式可直接对照。
            y_emb = y_emb + cfg.glyph_vec_scale * cfg.glyph_vec_out(e_glyph_vec)
    elif cfg.condition_fusion == "factorized_cat":
        # ref(Moyun) 式 concat 融合: 各向量因子 embedding 拼接 -> 联合 LN+Linear。
        # 见 __init__ 处说明。操作数 = {e_callig [, e_char] [, e_glyph_vec]}。
        # drop mask 同样复用 forward 顶部算好的那份 (y_callig_in / char_drop)。
        e_callig = cfg.embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens)
        _parts = [e_callig]
        if cfg.use_script_cond:
            if y_script is None:
                raise RuntimeError("[triple] use_script_cond=True 但 forward 未收到 y_script")
            if cfg.training and script_drop is not None:
                y_script = torch.where(script_drop,
                                       cfg.y_script_embedder.num_classes, y_script)
            _parts.append(cfg.y_script_embedder(y_script, False))
        if cfg.use_char_cond:
            if cfg.training and char_drop is not None:
                y_char = torch.where(char_drop, cfg.y_char_embedder.num_classes, y_char)
            _parts.append(cfg.y_char_embedder(y_char, False))
        if cfg.glyph_vec_cond and e_glyph_vec is not None:
            _parts.append(e_glyph_vec)
        # ★ 2026-09-17: 批次不一致时 `torch.cat` 只会抛
        #   "Sizes of tensors must match except in dimension 1"
        #   —— 完全看不出是哪个条件通路、各自多少，极难定位。
        #   这里显式检查并报出每个操作数的 batch，把模糊错误变成可诊断的。
        _bs = [int(p.shape[0]) for p in _parts]
        if len(set(_bs)) != 1:
            raise RuntimeError(
                f"[factorized_cat] 条件操作数 batch 不一致: {_bs} "
                f"(x batch={int(x.shape[0])}, g_tok="
                f"{None if g_tok is None else int(g_tok.shape[0])}, "
                f"y_callig={int(y_callig_in.shape[0])})。"
                f" 通常是 CFG 路径里 x 被复制成 2B 但某个条件通路没跟上。")
        if cfg.cond_ln is not None:
            # ★ split 归一化: 各操作数**独立** LayerNorm 后再 cat (docs 97 §3)。
            # 顺序必须与 __init__ 里 _dims 的构造顺序一致:
            # [e_callig, (e_char), (e_glyph_vec)]
            if len(_parts) != len(cfg.cond_ln):
                raise RuntimeError(
                    f"[factorized_cat/split] 操作数个数 {len(_parts)} 与 "
                    f"cond_ln 个数 {len(cfg.cond_ln)} 不一致 "
                    f"(use_char_cond={cfg.use_char_cond}, "
                    f"glyph_vec_cond={cfg.glyph_vec_cond})。")
            _parts = [ln(p) for ln, p in zip(cfg.cond_ln, _parts)]
        y_emb = cfg.cond_fusion(torch.cat(_parts, dim=-1))
        # ★ 2026-09-23 修复：改动 1 的 _style_branch 原先**只接在
        #   factorized_add 上**，而 v13/v15/v17 全部用 factorized_cat
        #   -> 改动 1 在这条真实路径上完全是**死代码**（无参数、无梯度、
        #      _style_branch 的返回值被丢弃）。必须在 cat 分支同样接上。
        #   注意 e_callig 就是 _parts[0]，直接用即可。
        y_emb = cfg.style_branch(e_callig, y_emb)
    elif cfg.condition_fusion == "xl_highdim":
        # XL 高维条件：与 factorized_add 相同的 4-way 可控 mask（CFG 需要 uncond 维度）。
        # drop mask 已在 forward 顶部计算 (与骨架风格注入共享同一份)。
        if cfg.training and char_drop is not None:
            y_char = torch.where(char_drop, cfg.y_char_embedder.num_classes, y_char)
        e_callig = cfg.embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens)
        e_char = cfg.y_char_embedder(y_char, False)
        y_emb = cfg.cond_fusion(torch.cat([e_callig, e_char], dim=-1)) * cfg.y_scale
        y_emb = cfg.style_branch(e_callig, y_emb)   # 922/80 改动 1 接线
    else:
        e_callig = cfg.y_callig_embedder(y_callig, cfg.training)
        e_char = cfg.y_char_embedder(y_char, cfg.training)
        y_concat = torch.cat([e_callig, e_char], dim=-1)
        y_emb = cfg.cond_fusion(y_concat)
        y_emb = cfg.style_branch(e_callig, y_emb)   # 922/80 改动 1 接线
    # ★ 2026-10-05 Time-Gate: 条件注入随 t 阶梯调制 (与 glyph_gate_t 同语义)。
    #   动机: 高噪声(t 大)给足条件信息, 低噪声(t 小)让模型自己收尾; 阶梯而非
    #   斜坡 (斜坡在 t=0.1 仍留 floor, 等于没关 —— v17 注释里同一教训)。
    _cgw = None
    if cfg.cond_gate_t > 0.0:
        _cgt = (t.float().flatten() / 1000.0).clamp(0, 1)
        _cgw = torch.where(_cgt >= cfg.cond_gate_t,
                           torch.ones_like(_cgt),
                           torch.full_like(_cgt, cfg.cond_gate_floor))
        _cgw = _cgw.to(y_emb.dtype).view(-1, 1)
        y_emb = y_emb * _cgw
    return y_emb, e_callig, _cgw


@dataclass(frozen=True)
class CfgForwardCfg:
    """`forward_with_cfg` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 第三步 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 本类不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ 两个字段是**绑定方法**（`forward` / `forward_with_2axis_cfg` ✓）；
      `use_char_cond` 的默认值**必须与原 `getattr(self, "use_char_cond", True)` 一致** ✓✓
      （工具 tools/measure_deps.py 会主动报出这类陷阱 ✓）。
    """
    forward: Any                # 原 self.forward（**绑定方法** ✓）
    forward_with_2axis_cfg: Any                # 原 self.forward_with_2axis_cfg（**绑定方法** ✓）
    image_channels: Any
    style_hier: Any
    y_callig_embedder: Any
    y_char_embedder: Any
    use_char_cond: Any                # 原 getattr(self, "use_char_cond", True) ✓ 默认必须为 True ✓

    @classmethod
    def from_model(cls, m):
        return cls(
                   forward=m.forward,
                   forward_with_2axis_cfg=m.forward_with_2axis_cfg,
                   image_channels=getattr(m, "image_channels", None),
                   style_hier=getattr(m, "style_hier", None),
                   y_callig_embedder=getattr(m, "y_callig_embedder", None),
                   y_char_embedder=getattr(m, "y_char_embedder", None),
                   use_char_cond=getattr(m, "use_char_cond", True),
        )


def forward_with_cfg(cfg, x, t, y_callig, y_char, cfg_scale, g=None, cfg_glyph=1.0,
                     w_inter=None, y_script=None, y_callig_raw=None, y_pair=None):
    """CFG 前向（**逐行等价**于 `DiT_2Cond.forward_with_cfg` ✓，2026-10-13 [审查 PR3] 第三步 ✓）。

    只把 `self.X` 换成 `cfg.X`、把 `getattr(self, "use_char_cond", True)` 换成 `cfg.use_char_cond` ✓；
    **未改动任何数值逻辑、未新增/移动任何 nn.Module** ✓
    ⇒ 由 eval + train 两个前向黄金 + `--cfg1` 等价检查共同裁决 ✓✓。
    """
    """经典 2 路 CFG（默认），或委托给双轴 CFG。

    S2: ``y_script`` / ``y_callig_raw`` / ``y_pair`` 需与 ``y_callig`` 一起复制，
    否则 hier 分支会拿到错误长度的条件（batch 不一致）。

    ★ 2026-09-17: 增加 `cfg_glyph` 参数。给了就走 `forward_with_2axis_cfg`
      （风格轴 × 内容轴各自独立），否则保持原有 2 路行为（**向后兼容**）。

    ⚠ 默认 2 路路径里 `g2 = cat([g, g])` —— **两半给同一个 g**，
      所以任何"只由 g 驱动"的通路（xattn / glyph_embedder）在
      `eps_cond − eps_uncond` 里**完全抵消，对 CFG 贡献恒为 0**。
      这是有意的（"CFG 只强化 callig 风格"），但会让组件结论失真：
      **xattn 这类通路在 cfg 引导下天然被绕过**（doc54 的 "xattn strict≈0" 即此）。
      要做内容轴引导必须走 `forward_with_2axis_cfg`，且训练时 `glyph_drop_prob > 0`。
    """
    if cfg_glyph is not None:
        # ★ 2026-10-13 [审查 P0]: 原来**漏传** y_script / y_callig_raw / y_pair ✗
        #   —— 与本方法 docstring 自己的要求相矛盾（S2 条件必须随 y_callig 一起复制,
        #   否则 hier/三表分支拿到 None -> 条件 batch 不一致或直接报错 ✗）。
        #   现全部透传 ✓（复制/拼接由 forward_with_2axis_cfg 内部按 4-pass 处理 ✓）
        return cfg.forward_with_2axis_cfg(
            x, t, y_callig, y_char,
            cfg_callig=cfg_scale, cfg_glyph=cfg_glyph, w_inter=w_inter, g=g,
            y_script=y_script, y_callig_raw=y_callig_raw, y_pair=y_pair)
    # Duplicate every sample: first copy conditional, second copy unconditional.
    original_bs = x.shape[0]
    # ★ 2026-09-17 诊断: 入口处各张量的 batch 必须一致，否则复制后仍不一致，
    #   最终在 factorized_cat 的 cat 里报一个看不出原因的 size 错。
    _in_bs = {"x": int(x.shape[0]), "t": int(t.shape[0]),
              "y_callig": int(y_callig.shape[0]), "y_char": int(y_char.shape[0])}
    if g is not None:
        _in_bs["g"] = int(g.shape[0])
    if len(set(_in_bs.values())) != 1:
        raise RuntimeError(
            f"[forward_with_cfg] 入口 batch 不一致: {_in_bs}。"
            f" 复制成 2B 后仍会不一致 -> 后面 cat 会报模糊的 size 错。"
            f" 调用方（sample_latents 等）必须给同一段切片。")
    x = torch.cat([x, x], dim=0)
    t = torch.cat([t, t], dim=0)
    y_callig = torch.cat([y_callig, y_callig], dim=0)
    y_char = torch.cat([y_char, y_char], dim=0)
    uncond_callig = torch.full_like(y_callig, cfg.y_callig_embedder.num_classes)
    y_callig_combined = torch.cat([y_callig[:original_bs], uncond_callig[original_bs:]], dim=0)
    if cfg.use_char_cond:
        uncond_char = torch.full_like(y_char, cfg.y_char_embedder.num_classes)
        y_char_combined = torch.cat([y_char[:original_bs], uncond_char[original_bs:]], dim=0)
    else:
        y_char_combined = y_char    # v10b: char 因子不存在, 值被 forward 忽略
    # 标准字形条件 g 始终全给(两半都用真实 g): 字形内容是正条件, CFG 只强化 callig 风格
    g2 = torch.cat([g, g], dim=0) if g is not None else None
    # S2: 三个新条件都要跟着复制成 2B，且**风格轴必须做 uncond 半置 null**。
    #   y_script     -> 两半都给真实书体（结构轴，与 g 同源；与 g2=cat([g,g]) 一致）
    #   y_callig_raw -> [real, null]  ← 不置 null 的话 uncond 半仍带真书家，CFG 直接失效
    #   y_pair       -> [real, null]  ← 同上；且 pair>=num_pairs 时残差自动为 0
    _dup = lambda v: torch.cat([v, v], dim=0) if v is not None else None
    _null_callig = cfg.y_callig_embedder.num_classes
    _null_pair = cfg.style_hier.num_pairs if cfg.style_hier is not None else 0
    _comb = lambda v, nv: (torch.cat(
        [v, torch.full_like(v, int(nv))], dim=0) if v is not None else None)
    model_out = cfg.forward(x, t, y_callig_combined, y_char_combined, g=g2,
                             y_script=_dup(y_script),
                             y_callig_raw=_comb(y_callig_raw, _null_callig),
                             y_pair=_comb(y_pair, _null_pair))
    if isinstance(model_out, tuple):
        model_out = model_out[0]  # skel_head 启用时 forward 返回 (主输出, skel_pred)，CFG 只取主输出
    # Apply CFG only on image latent channels (eps subspace), not canny/skel structure channels.
    eps, rest = model_out[:, :cfg.image_channels], model_out[:, cfg.image_channels:]
    cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    # ★ 2026-09-17: 原来写的是
    #     eps = torch.cat([half_eps, half_eps], dim=0)   # 白做一次 2B 分配
    #     out = torch.cat([eps, rest], dim=1)
    #     return out[:original_bs]                        # 又把后一半丢掉
    #   即先复制一份再截断 —— 纯浪费一次 2B×image_channels 的分配与拷贝。
    #   直接用 cond 那半的 rest(aux 通道取条件分支的值), 形状即 (B, C+rest)。
    #   语义完全一致 (返回行 = 前 original_bs 行)。
    out = torch.cat([half_eps, rest[:original_bs]], dim=1)
    return out


@dataclass(frozen=True)
class TwoAxisCfg:
    """`forward_with_2axis_cfg` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 第四步 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ `forward` 是**绑定方法** ✓；其余用 getattr 兜底 ✓（防惰性访问 ✗）。
    """
    forward: Any                # 原 self.forward（**绑定方法** ✓）
    image_channels: Any
    style_hier: Any
    y_callig_embedder: Any

    @classmethod
    def from_model(cls, m):
        return cls(
                   forward=m.forward,
                   image_channels=getattr(m, "image_channels", None),
                   style_hier=getattr(m, "style_hier", None),
                   y_callig_embedder=getattr(m, "y_callig_embedder", None),
        )


def forward_with_2axis_cfg(cfg, x, t, y_callig, y_char, cfg_callig, cfg_glyph, w_inter,
                           g=None, y_script=None, y_callig_raw=None, y_pair=None):
    """双轴 CFG 前向（**逐行等价**于 `DiT_2Cond.forward_with_2axis_cfg` ✓，[审查 PR3] 第四步 ✓）。

    只把 `self.X` 换成 `cfg.X` ✓；**未改动任何数值逻辑、未新增/移动任何 nn.Module** ✓
    ⇒ 由 eval + train 两个前向黄金 + `--cfg1` 共同裁决 ✓✓。
    """
    """双轴 CFG: 风格轴(书家) × 内容轴(骨架 g) 各自独立强度。

    ★ 2026-09-17 修正。**旧实现是错的**（且零调用者，是死代码）:
      它把 "glyph 轴" 定义在 **`y_char`**（字符 ID）上, 并且
      `g4 = cat([g, g, g, g])` —— **g 在 4 个 pass 里完全相同**。
      后果: ① 对 `no_char_cond=True` 的架构(内容来自 g), glyph 轴是 **no-op**;
            ② g 在 `eps_cond − eps_uncond` 里**仍然完全抵消**,
               所以 xattn / glyph_embedder 这类"只由 g 驱动"的通路对 CFG 贡献恒为 0。
      现在 glyph 轴改为**变化 g 本身**（置零 = "无骨架"分支）。

    4 个 pass 沿 batch 维拼成一次 forward:
      1. full    : (y_callig, g)     -> eps_full     两个条件都有
      2. style   : (y_callig, g=0)   -> eps_style    只有书家(风格), 无内容
      3. content : (null,     g)     -> eps_content  只有骨架(内容), 无风格
      4. uncond  : (null,     g=0)   -> eps_uncond   都没有

    Möbius / Product-of-Experts 组合:
      eps = eps_uncond
          + cfg_glyph  * (eps_content - eps_uncond)                 内容轴
          + cfg_callig * (eps_style   - eps_uncond)                 风格轴
          + w_inter    * (eps_full - eps_content - eps_style + eps_uncond)  交互项

    ⚠ **前提**: `g=0` 必须是模型训练时见过的条件 —— 即 `glyph_drop_prob > 0`。
      该 drop 的实现是 `g = g * keep`（整张样本置零），注释里明确写着
      "与 CFG 兼容: 丢弃时该样本等价于 uncond-g 分支"。
      若 `glyph_drop_prob == 0`，content 轴同样**未训练**，本函数会给出
      误导性结果（与 callig null 行是同一类问题）。
    """
    B = x.shape[0]
    x4 = torch.cat([x, x, x, x], dim=0)
    t4 = torch.cat([t, t, t, t], dim=0)

    null_c = torch.full_like(y_callig, cfg.y_callig_embedder.num_classes)
    yc4 = torch.cat([y_callig, y_callig, null_c, null_c], dim=0)
    # char 通路保持不变（我们 no_char_cond=True，该值被 forward 忽略）
    yg4 = torch.cat([y_char, y_char, y_char, y_char], dim=0)
    # S2: 风格轴的两个新条件同样按 [real, real, null, null] 拼 4 份。
    # 书体属于结构轴 -> 4 份都给真实值（与 g 同一处理）。
    _null_callig = cfg.y_callig_embedder.num_classes
    _null_pair = cfg.style_hier.num_pairs if cfg.style_hier is not None else 0
    if y_script is not None:
        ys4 = torch.cat([y_script] * 4, dim=0)
    else:
        ys4 = None
    if y_callig_raw is not None:
        _nr = torch.full_like(y_callig_raw, int(_null_callig))
        ycr4 = torch.cat([y_callig_raw, y_callig_raw, _nr, _nr], dim=0)
    else:
        ycr4 = None
    if y_pair is not None:
        _np = torch.full_like(y_pair, int(_null_pair))
        yp4 = torch.cat([y_pair, y_pair, _np, _np], dim=0)
    else:
        yp4 = None

    if g is not None:
        zero_g = torch.zeros_like(g)
        g4 = torch.cat([g, zero_g, g, zero_g], dim=0)
    else:
        g4 = None

    model_out = cfg.forward(x4, t4, yc4, yg4, g=g4,
                             y_script=ys4, y_callig_raw=ycr4, y_pair=yp4)
    if isinstance(model_out, tuple):
        model_out = model_out[0]

    eps4, rest4 = model_out[:, :cfg.image_channels], model_out[:, cfg.image_channels:]
    eps_full, eps_style, eps_content, eps_uncond = torch.split(eps4, B, dim=0)

    eps_guided = (
        eps_uncond
        + cfg_glyph * (eps_content - eps_uncond)
        + cfg_callig * (eps_style - eps_uncond)
        + w_inter * (eps_full - eps_content - eps_style + eps_uncond)
    )

    rest = rest4[:B]
    out = torch.cat([eps_guided, rest], dim=1)
    return out


@dataclass(frozen=True)
class StyleBranchCfg:
    """`_style_branch` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 通用抽取 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ 全部用 getattr 兜底 ✓（防**惰性访问** ✗）；陷阱项默认值与原文一致 ✓。
    """
    _style_out: Any
    style_gain: Any
    style_ln_mod: Any

    @classmethod
    def from_model(cls, m):
        return cls(
                   _style_out=getattr(m, "_style_out", None),
                   style_gain=getattr(m, "style_gain", None),
                   style_ln_mod=getattr(m, "style_ln_mod", None),
        )


def style_branch(cfg, e_callig, y_emb):
    """**逐行等价**于 `DiT_2Cond._style_branch` ✓（[审查 PR3] 通用抽取 ✓）。"""
    """922/80 改动 1：把风格向量经独立 LN + 独立增益**加**进 y_emb。

    动机（见 ``__init__`` 的 ``style_ln`` / ``style_gain_init`` 注释）：
    ``c = t_emb + y_emb`` 里 ``y_emb`` 只占 0.34–0.77，adaLN 共享投影后
    梯度被 ``t_emb`` 主导，且 zero-init 起步梯度 ∝ ``‖c‖`` -> 越弱越锁死。
    独立 LN 抹平 ``e_callig`` 的范数失衡（D5 实测 63 倍），
    独立增益把 ``y_emb`` 的起步幅度抬到与 ``t_emb`` 同量级，打破死锁。

    增益初值不硬编码，由 ``_init_style_gain`` 按实测范数标定
    （否则 LN 输出 19.1 vs t_emb 0.89 会直接过冲 21 倍）。

    返回**新的** y_emb（不改原张量，避免 in-place 影响 autograd 图）。
    未启用时（``style_ln_mod is None``）直接返回原值 —— 与旧 ckpt 逐位等价。

    ⚠ **`factorized_cat` / `xl_highdim` 模式下 `callig_proj` 是 None**
    （`__init__` 里显式置 None 以避免 DDP "参数未参与前向"报错，
    那些模式走 `cond_fusion` 自己融合）。所以这里**不能依赖 `callig_proj`**，
    改为用恒等 + LN：LN 后接一个 1x1 Linear 与"先 proj 再 LN"在表达能力上
    等价（都是逐样本的仿射+线性组合），且**不引入对现有投影的依赖**。
    这样三种 fusion 模式行为一致。
    """
    if cfg.style_ln_mod is None:
        return y_emb
    _s = cfg._style_out(e_callig)
    return y_emb + cfg.style_gain.to(y_emb.dtype) * _s.to(y_emb.dtype)


@dataclass(frozen=True)
class EmbedGlyphCfg:
    """`_embed_glyph` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 通用抽取 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ 全部用 getattr 兜底 ✓（防**惰性访问** ✗）；陷阱项默认值与原文一致 ✓。
    """
    _embed_callig: Any
    _embed_cond: Any
    callig_basis: Any
    callig_spatial_net: Any
    callig_style_ca: Any
    deform_skel: Any
    glyph_concat_input: Any
    glyph_embedder: Any
    glyph_scale: Any
    glyph_vec_cond: Any
    glyph_vec_pool: Any
    glyph_vec_proj: Any
    local_ca: Any
    local_ca_q: Any
    local_pos: Any
    lowrank_spatial: Any
    pos_embed: Any
    rope: Any
    script_film: Any
    spatial_film: Any
    use_glyph_cond: Any
    x_embedder: Any
    y_callig_embedder: Any
    zero_subtract_vector: bool = False

    @classmethod
    def from_model(cls, m):
        return cls(
                   _embed_callig=getattr(m, "_embed_callig", None),
                   _embed_cond=getattr(m, "_embed_cond", None),
                   callig_basis=getattr(m, "callig_basis", None),
                   callig_spatial_net=getattr(m, "callig_spatial_net", None),
                   callig_style_ca=getattr(m, "callig_style_ca", None),
                   deform_skel=getattr(m, "deform_skel", None),
                   glyph_concat_input=getattr(m, "glyph_concat_input", None),
                   glyph_embedder=getattr(m, "glyph_embedder", None),
                   glyph_scale=(m.effective_glyph_scale() if hasattr(m, 'effective_glyph_scale') else getattr(m, "glyph_scale", None)),
                   zero_subtract_vector=bool(getattr(m,'std_soft_cond',False) and not getattr(m,'std_soft_bias',True)),
                   glyph_vec_cond=getattr(m, "glyph_vec_cond", None),
                   glyph_vec_pool=getattr(m, "glyph_vec_pool", None),
                   glyph_vec_proj=getattr(m, "glyph_vec_proj", None),
                   local_ca=getattr(m, "local_ca", None),
                   local_ca_q=getattr(m, "local_ca_q", None),
                   local_pos=getattr(m, "local_pos", None),
                   lowrank_spatial=getattr(m, "lowrank_spatial", None),
                   pos_embed=getattr(m, "pos_embed", None),
                   rope=getattr(m, "rope", None),
                   script_film=getattr(m, "script_film", None),
                   spatial_film=getattr(m, "spatial_film", None),
                   use_glyph_cond=getattr(m, "use_glyph_cond", None),
                   x_embedder=getattr(m, "x_embedder", None),
                   y_callig_embedder=getattr(m, "y_callig_embedder", None),
        )


def embed_glyph(cfg, x, g, keep, y_callig_in, e_style, e_script, style_tokens, _gate):
    """**逐行等价**于 `DiT_2Cond._embed_glyph` ✓（[审查 PR3] 通用抽取 ✓）。"""
    """标准字形条件的嵌入与逐层注入准备（原 forward 内联 106 行 ✓）。

    包含: 形变骨架 DeformSkel -> glyph_embedder(含 std-soft 零输入修复) ->
    S2 通路 B1/B2/B3 -> 全局内容向量 e_glyph_vec。返回 (x, g_tok, e_glyph_vec)。
    """
    g_tok = None            # 标准字形 token；未启用时保持 None（逐层注入会检查）
    if cfg.rope:
        # 位置信息由 RoPE 在 attention 内部注入，不再加到残差流上。
        # 这样 token 幅度不随位置编码偏移，也天然支持不同 grid 的外推。
        pass
    else:
        x = x + cfg.pos_embed
    # ★ 形变标准骨架: g -> g'(书家习惯间架)。必须放在 g_tok 之前,
    #   这样 concat / adaLN 注入 / local_ca 三条通路都用同一个 g'。
    if cfg.deform_skel is not None and g is not None:
        _g2 = cfg.deform_skel(g, cfg._embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens))
        # ⚠ 条件被 drop 的样本（CFG uncond 分支）拿到的是 **null 风格向量** ->
        #   形变头会产出垃圾骨架, 等于往条件里注入噪声。
        #   实测: 10% 条件丢弃时 Deform loss 从离线的 0.167 涨到 0.37
        #   (0.9*0.167 + 0.1*2.0 ≈ 0.35, 吻合) —— 正是这个原因。
        #   -> 对丢弃样本**跳过形变**, 保持原骨架。
        # ⚠ 模型上**没有** num_calligraphers 属性（只是构造形参）——
        #   用 getattr(...,0) 会拿到 0, 让这段修复变成空操作。
        #   null 索引的真值在 y_callig_embedder.num_classes。
        _ncal = int(getattr(cfg.y_callig_embedder,
                            "num_classes", 0))
        if _ncal > 0:
            _kept = (y_callig_in < _ncal)
            if not bool(_kept.all()):
                g = torch.where(_kept.view(-1, 1, 1, 1), _g2, g)
            else:
                g = _g2
        else:
            g = _g2
    if cfg.use_glyph_cond and cfg.glyph_embedder is not None and g is not None:
        # 两类 glyph_embedder 最终都输出 (N,D,16,16) -> flat (N,256,D)。
        # ★★ 2026-10-12 修（v81 白化根因，专家定位）: **显式减去零输入响应**
        #   实测证据(probe2_glyph.py, v81 同配方):
        #     ‖encoder(0)‖ : 19.93(20k) → 34.11(35k) → 50.69(55k) → 72.07(115k)  = 3.6×
        #     内容/背景比  : 2.48 → 1.76 → 1.41 → 1.13（腰斩）
        #     同期生成墨量 : 0.174(35k，最佳) → 0.098(55k) → 0.073(115k)
        #     稀疏字不画 / 重墨字过墨（體 0.4948 vs GT 0.3559）→ 条件→墨迹映射失控
        #   机制: 编码器前两层 Conv2d 带 bias，而 `glyph_drop_prob` 默认 **0**
        #     ⇒ 训练中从未出现"无条件"状态 ⇒ 背景/偏置响应无人约束、自由膨胀
        #     ⇒ 注入主干的"空白画布特征"逐渐压过内容 ⇒ latent 空间结构消失
        #     ⇒ 解码成 DC ⇒ **白纸**（实测: latent 取均值 → 灰度 254.7 / 硬墨 0.0000）
        #   修法: feat = enc(g) - enc(0)，**不 detach 第二项** ⇒ g=0 ⟹ feat=0
        #     （即使编码器内部仍有 bias）。注意: 仅去掉 Conv bias 不够——膨胀来自深层。
        # ★ 2026-10-13 [审查 P1]: 零输入只算**一张**再广播 ✓ —— 编码器是 Conv2d+SiLU,

        #   无 BatchNorm/批间交互 ⇒ 单样本零图与整批零图逐位等价（审查实测 max|diff|=0 ✓）。

        #   ⚠ 梯度路径不变: 这里**不 detach** ✓（_g_zero 是常量, 无梯度需求）

        _g_zero = g.new_zeros((1, *g.shape[1:]))
        g_tok = (cfg.glyph_embedder(g) - cfg.glyph_embedder(_g_zero)).flatten(2).transpose(1, 2)
        if g_tok.shape[1] != x.shape[1]:
            raise ValueError(
                f"glyph token grid mismatch: g produced {g_tok.shape[1]} tokens but "
                f"image latent has {x.shape[1]}; std_soft_cond expects 1x64x64 input")
        if cfg.callig_style_ca is not None:
            # 书家化骨架(cross-attn): 骨架 token(+2D 位置) 在 K 个风格模态间
            # 内容寻址聚合, 产生"书家x字x位置"交互(结体差异)。
            # 注意 out_proj zero-init -> 注入从 0 平滑启动, 残差注入标准做法。
            # style_tokens 用 drop 后的 y_callig_in 查表: drop-callig 样本的
            # style token 全组来自 null_embed, 与 adaLN 分支一致。
            g_tok = cfg.callig_style_ca(g_tok, style_tokens)
        if cfg.callig_spatial_net is not None and cfg.callig_basis is not None:
            # 外挂(低秩): 书家向量 -> r 系数 x (r,256,D) 基图 -> 逐 token 加到骨架
            e_c_sp = cfg.y_callig_embedder(y_callig_in, False)
            _coef = cfg.callig_spatial_net(e_c_sp)                    # (N, r)
            g_tok = g_tok + torch.einsum("nr,rpd->npd", _coef, cfg.callig_basis)
        # ── S2 通路 B1：书体 FiLM（结构轴，全局 γ/β）────────────────────
        if cfg.script_film is not None and e_script is not None:
            g_tok = cfg.script_film(g_tok, e_script, keep)
        # ── 低秩逐位置风格×几何乘法 ─────────────────────────────────────
        # 用纯 e_callig, 不用拼了书体的 cfg._embed_cond(y_callig_in, e_style=e_style, style_tokens=style_tokens, e_script=e_script): 书体已在 g 里。
        # 放在 script_film 之后、spatial_film 之前, 三者互不覆盖。
        if cfg.lowrank_spatial is not None:
            g_tok = cfg.lowrank_spatial(g_tok, cfg._embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens), cfg.local_pos, keep)
        # ── S2 通路 B2：逐位置局部 FiLM（Phase 1）───────────────────────
        if cfg.spatial_film is not None:
            g_tok = cfg.spatial_film(g_tok, cfg._embed_cond(y_callig_in, e_style=e_style, style_tokens=style_tokens, e_script=e_script), cfg.local_pos, keep)
        # ── S2 通路 B3：局部风格-骨架 cross-attn（q="g" 时在条件侧堆叠）──
        # 语义："先按风格重组骨架，再注入主干"。比 q="x" 更安全、可回退。
        if cfg.local_ca is not None and g_tok is not None and cfg.local_ca_q == "g":
            _cond = cfg._embed_cond(y_callig_in, e_style=e_style, style_tokens=style_tokens, e_script=e_script)
            for _lc in cfg.local_ca:
                g_tok = _lc(g_tok, g_tok, _cond, cfg.local_pos, keep)
        if keep is not None:
            # ⚠ 丢弃语义保护: style 注入会给零骨架加非零风格输出, 会把
            # "uncond-g 分支"(drop 的样本)重新变成有条件 —— 必须**在风格调制
            # 之后**把被丢弃样本的 g_tok 重新置零, 保持 drop 分支纯净。
            g_tok = g_tok * keep.view(-1, 1, 1).to(g_tok.dtype)
        if not cfg.glyph_concat_input:
            # 拼接模式不再把骨架加到残差上。g_tok 仍留给交叉注意力当 K/V。
            x = x + cfg.glyph_scale * (_gate * g_tok if _gate is not None else g_tok)

    # ── g 的全局内容向量 (进入条件向量 c) ────────────────────────────────
    # 把 g_tok 的 256 个 token 池化成一个向量, 作为 concat/add 的第二个操作数。
    # 池化源用**已做 drop 掩码后**的 g_tok, 保证 drop-g 样本的该向量也为零
    # (与"该样本没有 g 条件"的语义一致)。
    e_glyph_vec = None
    if cfg.glyph_vec_cond:
        if _gate is not None and g_tok is not None:
            # 池化进 adaLN 的那一路也要乘, 否则只弱了输入残差, 条件向量里
            # 的骨架还是全强度。
            g_tok = _gate * g_tok
        if g_tok is not None:
            _pooled = (g_tok.amax(dim=1) if cfg.glyph_vec_pool == "max"
                       else g_tok.mean(dim=1))
        else:
            # 完全没有 g 时走零向量 —— 仍经过 proj, 保证 DDP 下该分支参数
            # 参与前向 (否则报 "parameters that didn't receive grad")。
            _pooled = torch.zeros(x.shape[0], cfg.x_embedder.hidden_size,
                                  device=x.device, dtype=x.dtype)
        e_glyph_vec = cfg.glyph_vec_proj(_pooled)
        if cfg.zero_subtract_vector:
            e_glyph_vec = e_glyph_vec-cfg.glyph_vec_proj(torch.zeros_like(_pooled))
    return x, g_tok, e_glyph_vec


@dataclass(frozen=True)
class InitWeightsCfg:
    """`initialize_weights` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 通用抽取 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ 全部用 getattr 兜底 ✓（防**惰性访问** ✗）；陷阱项默认值与原文一致 ✓。
    """
    _init_style_gain: Any
    _style_gain_init: Any
    apply: Any
    blocks: Any
    callig_spatial_net: Any
    callig_style_ca: Any
    deform_skel: Any
    final_layer: Any
    glyph_concat_input: Any
    glyph_injections: Any
    local_ca: Any
    lowrank_spatial: Any
    pos_embed: Any
    script_film: Any
    skel_head: Any
    skel_head_enabled: Any
    spatial_film: Any
    style_ada_rank: Any
    style_gain: Any
    style_hier: Any
    t_embedder: Any
    x_embedder: Any
    y_callig_embedder: Any
    y_char_embedder: Any

    @classmethod
    def from_model(cls, m):
        return cls(
                   _init_style_gain=getattr(m, "_init_style_gain", None),
                   _style_gain_init=getattr(m, "_style_gain_init", 1.0),
                   apply=getattr(m, "apply", None),
                   blocks=getattr(m, "blocks", None),
                   callig_spatial_net=getattr(m, "callig_spatial_net", None),
                   callig_style_ca=getattr(m, "callig_style_ca", None),
                   deform_skel=getattr(m, "deform_skel", None),
                   final_layer=getattr(m, "final_layer", None),
                   glyph_concat_input=getattr(m, "glyph_concat_input", None),
                   glyph_injections=getattr(m, "glyph_injections", None),
                   local_ca=getattr(m, "local_ca", None),
                   lowrank_spatial=getattr(m, "lowrank_spatial", None),
                   pos_embed=getattr(m, "pos_embed", None),
                   script_film=getattr(m, "script_film", None),
                   skel_head=getattr(m, "skel_head", None),
                   skel_head_enabled=getattr(m, "skel_head_enabled", False),
                   spatial_film=getattr(m, "spatial_film", None),
                   style_ada_rank=getattr(m, "style_ada_rank", 0),
                   style_gain=getattr(m, "style_gain", None),
                   style_hier=getattr(m, "style_hier", None),
                   t_embedder=getattr(m, "t_embedder", None),
                   x_embedder=getattr(m, "x_embedder", None),
                   y_callig_embedder=getattr(m, "y_callig_embedder", None),
                   y_char_embedder=getattr(m, "y_char_embedder", None),
        )


def initialize_weights(cfg, ):
    """**逐行等价**于 `DiT_2Cond.initialize_weights` ✓（[审查 PR3] 通用抽取 ✓）。"""
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            # 严禁重置形变网络权重
            if cfg.deform_skel is not None and any(module is m for m in cfg.deform_skel.modules()):
                return
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    cfg.apply(_basic_init)

    pos_embed = get_2d_sincos_pos_embed(cfg.pos_embed.shape[-1], int(cfg.x_embedder.num_patches ** 0.5))
    cfg.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    w = cfg.x_embedder.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(cfg.x_embedder.proj.bias, 0)
    if cfg.glyph_concat_input:
        # 后 4 个输入通道是骨架。置 0, 第 0 步骨架对输出没有贡献。
        c0 = w.shape[1] - 4
        if c0 > 0:
            w[:, c0:].zero_()

    nn.init.normal_(cfg.y_callig_embedder.embedding_table.weight, std=0.02)
    # v10b: use_char_cond=False 时 char 侧整体不存在
    # IDSCharEmbedder 用 comp_embedding 而非 embedding_table
    if cfg.y_char_embedder is None:
        pass    # v10b: 无 char 侧
    elif cfg.y_char_embedder is not None and hasattr(cfg.y_char_embedder, 'comp_embedding'):
        nn.init.normal_(cfg.y_char_embedder.comp_embedding.weight, std=0.02)
        if cfg.y_char_embedder.null_embed is not None:
            nn.init.normal_(cfg.y_char_embedder.null_embed, std=0.02)
        if cfg.y_char_embedder.fallback_embed is not None:
            nn.init.normal_(cfg.y_char_embedder.fallback_embed, std=0.02)
    elif hasattr(cfg.y_char_embedder, 'char_table'):
        # 标准字形 DINO 冻结表：不重新初始化（保持标准字形特征）。
        # 只初始化可学习的 CFG null token。
        if cfg.y_char_embedder.null_embed is not None:
            nn.init.normal_(cfg.y_char_embedder.null_embed, std=0.02)
    else:
        nn.init.normal_(cfg.y_char_embedder.embedding_table.weight, std=0.02)

    if cfg.skel_head_enabled and cfg.skel_head is not None:
        nn.init.zeros_(cfg.skel_head[-1].weight)
        nn.init.zeros_(cfg.skel_head[-1].bias)

    nn.init.normal_(cfg.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(cfg.t_embedder.mlp[2].weight, std=0.02)

    for block in cfg.blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    nn.init.constant_(cfg.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(cfg.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(cfg.final_layer.linear.weight, 0)
    nn.init.constant_(cfg.final_layer.linear.bias, 0)

    # 标准字形的逐层注入必须**保持 zero-init**。
    #
    # ZeroAdaLNInjection 在构造时已把 proj 的 weight/bias 置零，但上面
    # 的 ``cfg.apply(_basic_init)`` 会把**所有** nn.Linear（含这些 proj）
    # 重新初始化成 xavier_uniform —— 于是注入不再是恒等映射，
    # warm-start 语义被破坏：从已训练 ckpt 续跑时，新加的注入层会给
    # 残差流注入随机扰动，等价于在训练好的模型上叠加噪声。
    #
    # 这里显式重新置零，恢复 ``out = x*(1+s)+t`` 中 s=t=0 的恒等起点。
    if cfg.glyph_injections is not None:
        for _inj in cfg.glyph_injections:
            if hasattr(_inj, "out_proj"):      # ZeroCrossAttention
                nn.init.zeros_(_inj.out_proj.weight)
                nn.init.zeros_(_inj.out_proj.bias)
            else:                              # ZeroAdaLNInjection
                nn.init.zeros_(_inj.proj.weight)
                nn.init.zeros_(_inj.proj.bias)

    # callig 空间化最后一层 (to_spatial) 同样恢复 zero-init:
    # s_callig=0 初始恒等, 从已训 ckpt 续跑时不注入随机扰动。
    if cfg.callig_spatial_net is not None:
        nn.init.zeros_(cfg.callig_spatial_net[-1].weight)
        nn.init.zeros_(cfg.callig_spatial_net[-1].bias)

    # callig 风格 cross-attn 的 out_proj 同样恢复 zero-init (resume 恒等)。
    if cfg.callig_style_ca is not None:
        nn.init.zeros_(cfg.callig_style_ca.out_proj.weight)
        nn.init.zeros_(cfg.callig_style_ca.out_proj.bias)

    # ── S2 新模块：同样必须恢复 zero-init（_basic_init 会把它们 xavier 掉）──
    if cfg.script_film is not None:
        nn.init.zeros_(cfg.script_film.film.weight)
        nn.init.zeros_(cfg.script_film.film.bias)
    if cfg.spatial_film is not None:
        nn.init.zeros_(cfg.spatial_film.net[-1].weight)
        nn.init.zeros_(cfg.spatial_film.net[-1].bias)
    if cfg.lowrank_spatial is not None:
        # 不能 zeros_: 全零会让 u_i 的下游梯度为 0, P/Q 一步不动。
        cfg.lowrank_spatial.reset_output()
    if cfg.local_ca is not None:
        for _lc in cfg.local_ca:
            # _basic_init 会把新模块的 Linear 重新 xavier 掉, 限幅初始值必须再设一次。
            if hasattr(_lc, "reset"):
                _lc.reset()          # GlyphQuery
            else:                    # legacy LocalStyleGlyphAdapter: 它自己那套
                nn.init.zeros_(_lc.out_proj.weight)
                nn.init.zeros_(_lc.out_proj.bias)
                nn.init.zeros_(_lc.style_qk[-1].weight)
                nn.init.zeros_(_lc.style_qk[-1].bias)
    # pair 残差表保持 zero-init（残差语义：step 0 退化为纯主效应）
    if cfg.style_hier is not None and str(
            getattr(cfg.style_hier, "_pair_init", "zero")) == "zero":
        nn.init.zeros_(cfg.style_hier.E_pair.weight)

    # ── 922/80 改动 2：风格专用 adaLN 支路的 W_up 必须**重新初始化** ──
    # ⚠ 极易漏：`_basic_init` 对**所有** nn.Linear 做 xavier_uniform_，
    #   会把 DiTBlock/FinalLayer.__init__ 里刚设好的初值**整片冲掉**。
    #   （同样的坑此前已在 script_film / spatial_film / local_ca 上踩过。）
    #
    # ⚠⚠ 这里**不能**用 zeros_：W_up=0 会让 ∂L/∂W_up ≡ 0（见 T5 实测），
    #    支路永远学不动。必须用 M.STYLE_ADA_INIT_STD 的小随机初值，
    #    让梯度从头就 ∝ ‖W_up‖ ≠ 0。
    if cfg.style_ada_rank > 0:
        for _m in list(cfg.blocks) + [cfg.final_layer]:
            if getattr(_m, "style_ada_up", None) is not None:
                # ⚠ v85A: 必须带上 gain 标志 ✗ —— 否则这里会把"非零 bias"
                #   又抹回 0 ✓（实测 up.bias std = 0.0 ⇒ Δ 可能恒为 0 ⇒
                #   ∂L/∂gain = δ·Δ ≡ 0 ⇒ 支路永远启动不了 ✗）
                _m._style_ada_init_up(
                    _m.style_ada_up,
                    nonzero_bias=getattr(_m, "_style_ada_gain_on", False))

    # ── 922/80 改动 1：style_gain 初值按实测范数**自动标定** ──
    # 必须放在最后：依赖 t_embedder / y_callig_embedder / style_ln_mod 都已建好。
    # 传 style_gain_init=None（或负数）走自动标定，否则用显式值。
    if cfg.style_gain is not None:
        _gi = cfg._style_gain_init
        if _gi is None or float(_gi) < 0:
            cfg._init_style_gain()
        else:
            with torch.no_grad():
                cfg.style_gain.fill_(float(_gi))
    # ★ 2026-10-14 [重构 P1-1] 参数锁由 `apply_post_construction(model, ...)` 设置 ✓
    #   （本函数只拿到 cfg 视图 ✗ 没有 model ✗ —— 见 src/eval/model_io.py 末尾 ✓）


@dataclass(frozen=True)
class ForwardCfg:
    """`forward` 用到的**全部** `self.*`（2026-10-13 [审查 PR3] 通用抽取 ✓）。

    审查: "只移动逻辑, 不移动权重注册位置" ✓ ⇒ 不注册模块 ⇒ **state_dict 键不变** ✓✓。
    ⚠ 全部用 getattr 兜底 ✓（防**惰性访问** ✗）；陷阱项默认值与原文一致 ✓。
    """
    _cond_drop_masks: Any
    _embed_callig: Any
    _embed_cond: Any
    _embed_glyph: Any
    _fuse_condition: Any
    _inj_map: Any
    _local_ca_map: Any
    blocks: Any
    callig_multi_style_k: Any
    cond_router: Any
    ctx_pos_g: Any
    final_layer: Any
    glyph_concat_input: Any
    glyph_gate_floor: Any
    glyph_gate_t: Any
    glyph_injections: Any
    glyph_vec_cond: Any
    local_ca: Any
    local_ca_q: Any
    local_pos: Any
    n_style_token: Any
    rope: Any
    rope_cos: Any
    rope_sin: Any
    skel_head: Any
    skel_head_enabled: Any
    style_ada_rank: Any
    style_ctx_every_layer: Any
    style_hier: Any
    style_proj: Any
    style_role: Any
    t_embedder: Any
    unpatchify: Any
    use_glyph_cond: Any
    x_embedder: Any
    y_callig_embedder: Any
    y_script_embedder: Any
    writer_glyph: Any

    @classmethod
    def from_model(cls, m):
        return cls(
                   _cond_drop_masks=getattr(m, "_cond_drop_masks", None),
                   _embed_callig=getattr(m, "_embed_callig", None),
                   _embed_cond=getattr(m, "_embed_cond", None),
                   _embed_glyph=getattr(m, "_embed_glyph", None),
                   y_script_embedder=getattr(m, "y_script_embedder", None),
                   writer_glyph=getattr(m, "writer_glyph", None),
                   _fuse_condition=getattr(m, "_fuse_condition", None),
                   _inj_map=getattr(m, "_inj_map", None),
                   _local_ca_map=getattr(m, "_local_ca_map", None),
                   blocks=getattr(m, "blocks", None),
                   callig_multi_style_k=getattr(m, "callig_multi_style_k", None),
                   cond_router=getattr(m, "cond_router", None),
                   ctx_pos_g=getattr(m, "ctx_pos_g", None),
                   final_layer=getattr(m, "final_layer", None),
                   glyph_concat_input=getattr(m, "glyph_concat_input", None),
                   glyph_gate_floor=getattr(m, "glyph_gate_floor", None),
                   glyph_gate_t=getattr(m, "glyph_gate_t", None),
                   glyph_injections=getattr(m, "glyph_injections", None),
                   glyph_vec_cond=getattr(m, "glyph_vec_cond", None),
                   local_ca=getattr(m, "local_ca", None),
                   local_ca_q=getattr(m, "local_ca_q", None),
                   local_pos=getattr(m, "local_pos", None),
                   n_style_token=getattr(m, "n_style_token", None),
                   rope=getattr(m, "rope", None),
                   rope_cos=getattr(m, "rope_cos", None),
                   rope_sin=getattr(m, "rope_sin", None),
                   skel_head=getattr(m, "skel_head", None),
                   skel_head_enabled=getattr(m, "skel_head_enabled", None),
                   style_ada_rank=getattr(m, "style_ada_rank", 0),
                   style_ctx_every_layer=getattr(m, "style_ctx_every_layer", False),
                   style_hier=getattr(m, "style_hier", None),
                   style_proj=getattr(m, "style_proj", None),
                   style_role=getattr(m, "style_role", None),
                   t_embedder=getattr(m, "t_embedder", None),
                   unpatchify=getattr(m, "unpatchify", None),
                   use_glyph_cond=getattr(m, "use_glyph_cond", None),
                   x_embedder=getattr(m, "x_embedder", None),
                   y_callig_embedder=getattr(m, "y_callig_embedder", None),
        )


def forward(cfg, x, t, y_callig, y_char, return_intermediate_layer, return_intermediate_layers, g, y_script, y_callig_raw, y_pair):
    """**逐行等价**于 `DiT_2Cond.forward` ✓（[审查 PR3] 通用抽取 ✓）。"""
    """
    Forward pass of DiT_2Cond.
    x: (N, C, H, W) noisy latents
    t: (N,) timesteps
    y_callig, y_char: (N,) condition IDs
    g: 旧路径为 (N,4,32,32) latent；std-soft 为 (N,1,64,64) raw mask；None=不使用
    return_intermediate_layer: int block index (e.g. 8) whose patch features to return for REPA.
                               When set, returns (output, intermediate_feats) as a tuple.

    S2 新增条件（全部可选，None = 该层缺失 / 走旧路径）:
    y_script:     (N,) 书体 id（楷/行/隶）→ E_script
    y_callig_raw: (N,) 书家**连续索引**（不是 pair id）→ 主效应表
    y_pair:       (N,) (书家×书体) pair id → E_pair 残差
    """
    # ── 标准字形条件的随机丢弃（仅训练时）────────────────────────────
    #
    # 为什么要 drop：
    # 1. 防止「门控」。标准字形 latent 与 DINO CLS 都由字 ID 推出，存在冗余。
    #    std-skel 实验已证明：冗余条件会被 ControlNet 学成恒等门控掉
    #    （SSIM 0.494≈base，SkelIoU 0.015，12 个评测点全平）。
    #    随机丢弃使 g 不可预测，模型无法依赖「反正能从字 ID 推出来」而忽略它。
    # 2. 覆盖现实：fame 上 v2 字典只覆盖 53.1%，草/篆（46.9%）本来就没有
    #    标准字形，推理时 g 只能是零。训练时模拟这种缺失，让模型学会
    #    「g 有效时跟随，g 为零时依赖书家+字条件」。
    # 3. 与 CFG 兼容：丢弃时该样本等价于 uncond-g 分支。
    #
    # 注意：以**整张样本**为单位丢弃（而非逐像素），与「某字没有标准字形」
    # 的真实情况一致。
    # ★ 2026-10-13: 原 46 行内联条件遮罩已抽成 _cond_drop_masks ✓（返回值由**输出穷举分析**决定, 防漏交 ✗）
    g, keep, callig_drop, char_drop, script_drop, y_callig_in = cfg._cond_drop_masks(g, y_callig)

    # ── v15 多模态风格 token：一次查表, 骨架注入与 adaLN 融合共享 ──────────
    # (B, K, D); drop 已由 y_callig_in 携带 (null 标签 -> null_embed 整组替换)
    style_tokens = None
    if cfg.callig_multi_style_k > 0:
        style_tokens = cfg.y_callig_embedder(y_callig_in, False)

    # ── S2：三层语义分解 ─────────────────────────────────────────────────
    #   e_style  = 书家主效应(既有表) + α · E_pair[pair]
    #   e_script = E_script[script]
    # 主效应表用 **y_callig_raw（书家连续索引）**，pair 残差用 **y_pair**。
    # ⚠ 两者不是同一个 id：v14/v15 里 y_callig 装的是 pair_id(87)，
    #    hier 模式下主效应表只有 45 行，必须用 raw，否则越界/串书家。
    e_style = None
    e_script = None
    if cfg.style_hier is not None:
        _base_ids = y_callig_raw if y_callig_raw is not None else y_callig
        _base_in = (torch.where(callig_drop, cfg.y_callig_embedder.num_classes,
                                _base_ids)
                    if callig_drop is not None else _base_ids)
        _base = cfg.y_callig_embedder(_base_in, False)
        e_style, e_script = cfg.style_hier(_base, y_pair, y_script, callig_drop)

    # ★ 2026-10-13: 原两个闭包升格为方法（逻辑见 _embed_callig / _embed_cond ✓）。
    #   这里保留一行转发 ⇒ **调用点完全不变**（最小改动 ✓）；升格后可被
    #   _fuse_condition 等其它方法复用, 也便于单独测试 ✓
    def _e_callig():
        return cfg._embed_callig(y_callig_in, e_style=e_style, style_tokens=style_tokens)

    def _e_cond():
        return cfg._embed_cond(y_callig_in, e_style=e_style,
                                style_tokens=style_tokens, e_script=e_script)

    if cfg.glyph_concat_input:
        # g 在上面的 drop 里已经被整样本置零。拼在图像后面,
        # 第一层卷积的后 4 个通道初始为 0, 第 0 步等于没有骨架。
        g_in = torch.zeros_like(x) if g is None else g
        x = cfg.x_embedder(torch.cat([x, g_in], dim=1))
    else:
        x = cfg.x_embedder(x)  # (N, T, D)
    # 时间门: 训练和采样传进来的 t 都乘过 TIME_SCALE=1000, 这里除回来。
    # 阶梯, 不是斜坡。t≥glyph_gate_t 时为 1; 低于它时三路骨架条件
    # 直接锁在 glyph_gate_floor。斜坡会让 t=0.1 仍有 0.36, 等于没关。
    _gate = None
    if cfg.glyph_gate_t > 0 and cfg.use_glyph_cond:
        _tf = (t.float().flatten() / 1000.0).clamp(0, 1)
        _w = torch.where(
            _tf >= cfg.glyph_gate_t,
            torch.ones_like(_tf),
            torch.full_like(_tf, cfg.glyph_gate_floor))
        _gate = _w.to(x.dtype).view(-1, 1, 1)
    # ★ 2026-10-13: 原 106 行内联字形嵌入已抽成 _embed_glyph ✓
    x, g_tok, e_glyph_vec = cfg._embed_glyph(x, g, keep, y_callig_in, e_style, e_script, style_tokens, _gate)
    writer_delta = None
    if cfg.writer_glyph is not None and g_tok is not None:
        if y_script is None:
            raise ValueError('writer_glyph needs explicit script IDs')
        script_in = (torch.where(script_drop, cfg.y_script_embedder.num_classes, y_script)
                     if script_drop is not None else y_script)
        style = torch.cat([_e_callig(), cfg.y_script_embedder(script_in, False)], dim=-1)
        writer_keep = y_callig_in < cfg.y_callig_embedder.num_classes
        writer_delta = cfg.writer_glyph.residual(g_tok, style, g, writer_keep, keep)
        if _gate is not None:
            writer_delta = writer_delta * _gate

    t_emb = cfg.t_embedder(t)               # (N, D)

    # ★ 2026-10-13: 原 115 行内联条件融合已抽成 _fuse_condition ✓
    y_emb, e_callig, _cgw = cfg._fuse_condition(
        t, x, y_callig_in, y_callig, y_char, y_script, char_drop, script_drop,
        e_glyph_vec, g_tok, e_style, style_tokens)
    c = t_emb + y_emb                        # (N, D)  —— 旧行为/路由白名单层的条件

    # ---- 922/80 改动 2: 风格专用支路的输入 ----
    # 直接喂**原始** e_callig（不是 y_emb），理由：
    #   e_callig 是"纯风格"信号，不含 t、不含 char/glyph 产物；
    #   若喂 y_emb 则又走回共享混合的旧问题（其中 t 间接进入不了，但
    #   char/glyph 会串味，且 factorized_cat 下 y_emb 已被投影混合）。
    # 用独立 LN 归一化，范数失衡（D5 实测 63 倍）不会直接传到 W_up。
    # zero-init -> 关闭时 c_style 不参与任何计算（也不建参数）。
    c_style = None
    if cfg.style_ada_rank > 0:
        c_style = e_callig
    # Time-Gate 同时作用于风格专用支路 (它同样是"条件注入")。
    if _cgw is not None and c_style is not None:
        c_style = c_style * _cgw

    rope = (cfg.rope_cos, cfg.rope_sin) if cfg.rope else None

    intermediate_feats = None
    # 多层 REPA 捕获 (统一): return_intermediate_layers=(8,11) -> dict {8:feats, 11:feats}
    _repa_layers = None
    if return_intermediate_layers is not None:
        _repa_layers = (return_intermediate_layers
                        if isinstance(return_intermediate_layers, (list, tuple))
                        else (return_intermediate_layers,))
        intermediate_feats = {}
    # 兼容旧单层接口
    _repa_single = None
    if return_intermediate_layer is not None:
        _repa_single = int(return_intermediate_layer)
    # 预计算好的映射（__init__ 里建），不再每步重建
    _inj = {}
    if cfg.glyph_injections is not None and g_tok is not None:
        _inj = cfg._inj_map

    # ── 逐层注入的 context ───────────────────────────────────────────────
    # 默认 = 骨架 token(GlyphStyleCrossAttn 模式下位置在下面显式加)。
    # style_token_n>0 时拼接风格 token: 让**每一层**的 attention 都能直接
    # 看到书家风格, 而不是只通过"书家化骨架"间接进入(会被深层稀释)。
    # 每个 x 位置(query)因此可同时寻址: 局部字形(空间对应) + 书家风格。
    # 逐层注入是第三路, 不经过 glyph_scale。g_tok 在上面已经乘过门
    # (开了 glyph_vec_cond 时); 没开时在这里乘, 避免这一路漏掉。
    if _gate is not None and not cfg.glyph_vec_cond and g_tok is not None:
        g_tok = _gate * g_tok
    inject_ctx = g_tok
    if (cfg.style_ctx_every_layer
            and g_tok is not None and style_tokens is not None):
        # v15c: 骨架 token(+2D 位置) 与 K 个风格 token(+可学习 role) 拼成
        # 每层 xattn 的 K/V —— 风格在**每一层、每个空间位置**都可直接寻址。
        _Ng = g_tok.shape[1]
        inject_ctx = torch.cat(
            [g_tok + cfg.ctx_pos_g[:, :_Ng],
             style_tokens + cfg.style_role.unsqueeze(0)], dim=1)
    elif cfg.style_proj is not None and g_tok is not None:
        _B = g_tok.shape[0]
        _D = g_tok.shape[-1]
        _Ng = g_tok.shape[1]
        _style = cfg.style_proj(e_callig).view(
            _B, cfg.n_style_token, _D) + cfg.style_role.unsqueeze(0)
        # 骨架 token 加 2D sincos 位置(空间对应), 风格 token 用可学习 role
        inject_ctx = torch.cat([g_tok + cfg.ctx_pos_g[:, :_Ng], _style], dim=1)

    # ── S2 通路 B3（q="x"）：局部风格-骨架 adapter 插在**主干 block 之间** ──
    # 语义："去噪中的画布，在风格条件下向局部骨架询问书写证据"。
    # 比 q="g" 更标准，但改动更深入主干、成本略高。默认走 q="g"。
    _lc_map = {}
    _e_cond_local = None
    if (cfg.local_ca is not None and cfg.local_ca_q == "x"
            and g_tok is not None):
        _lc_map = cfg._local_ca_map
        _e_cond_local = _e_cond()

    # ★ [2026-10-06] 三表条件的分层路由：白名单层用 c（含 y_emb*scale），
    #   其余层只用 t_emb —— 条件在那几层**完全不参与计算**（不只是"少注入"）。
    _router = cfg.cond_router
    # ★★ v85A (2026-10-13) 关键语义修正（方案 §二.4）:
    #   原实现里 `else` 分支把 `c_style=None` 一并切断 ✗ ⇒ 若风格层不在白名单内,
    #   这条支路**白建**（参数存在但永远收不到输入 ✗）。改为：内容条件仍按白名单路由 ✓，
    #   而**书家风格走独立通路** ✓ —— 这正是"独立书家风格调制"的字面含义 ✓。
    _style_layers = getattr(cfg, "_style_ada_layers", None)
    if _style_layers is None:
        # 兜底: 从 block 自身推断（旧对象/未经过新构造路径时 ✓）
        _style_layers = tuple(i for i, b in enumerate(cfg.blocks)
                              if getattr(b, "style_ada_up", None) is not None)
    _style_layers = set(_style_layers)
    for i, block in enumerate(cfg.blocks):
        if _router is None:
            c_main = c
        elif _router.allows(i):
            c_main = _router.cond(i, t_emb, y_emb)
        else:
            c_main = t_emb
        x = block(x, c_main, rope=rope,
                  c_style=(c_style if i in _style_layers else None))
        if writer_delta is not None and i in cfg.writer_glyph.at:
            x = cfg.writer_glyph.inject(x, writer_delta, i)
        if _repa_layers is not None and i in _repa_layers:
            intermediate_feats[i] = x
        elif _repa_single is not None and i == _repa_single:
            intermediate_feats = x
        if i in _inj:
            # x = x*(1+s) + t，s/t 由 g_tok 经 zero-init Linear 产出。
            # init 时恒等；梯度上 ∂out/∂g_tok = W = 0，因此这条路径
            # 初期不给 glyph_embedder 梯度 —— 但输入层的
            # x = x + glyph_scale * g_tok（glyph_scale=0.4 非零）
            # 已提供直通梯度，故 glyph_embedder 从 step 0 即可学习。
            x = cfg.glyph_injections[_inj[i]](x, inject_ctx)
        if i in _lc_map:
            x = cfg.local_ca[_lc_map[i]](x, g_tok, _e_cond_local,
                                          cfg.local_pos, keep)

    # 骨架头：从 final_layer 前的 block 输出特征并行解码 latent 骨架 (N,1,32,32)
    skel_pred = None
    if cfg.skel_head_enabled and cfg.skel_head is not None:
        skel_n = cfg.skel_head(x)                       # (N, T, p*p)，单通道 patch 值
        B_, T_, PP = skel_n.shape
        h_ = int(T_ ** 0.5)
        p_ = cfg.x_embedder.patch_size[0]
        # 完全镜像主 head 的 unpatchify（C=1）：
        # (B,H*W,p*p) -> (B,H,W,p,p,1) -> einsum 'nhwpqc->nchpwq' -> (B,1,H,W,p,p) -> (B,1,H*p,W*p)
        skel5 = skel_n.reshape(B_, h_, h_, p_, p_, 1)
        skel5 = torch.einsum('nhwpqc->nchpwq', skel5)
        skel_pred = skel5.reshape(B_, 1, h_ * p_, h_ * p_)

    x = cfg.final_layer(x, c, c_style=c_style)
    x = cfg.unpatchify(x)
    if cfg.skel_head_enabled and skel_pred is not None:
        # 返回 (主输出, skel_pred)；gaussian_diffusion.training_losses 会把第二元素
        # 当作 intermediate_feats 存入 loss_dict['intermediate_feats']
        return x, skel_pred
    if _repa_layers is not None:
        return x, intermediate_feats   # dict {layer: feats} (多层 REPA)
    if return_intermediate_layer is not None:
        return x, intermediate_feats
    return x
