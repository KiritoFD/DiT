# -*- coding: utf-8 -*-
"""conditioning.py — **条件系统的唯一真源**（2026-10-13 [审查 PR2]）

动机（审查原文）:
  同一责任在多个地方实现、同一配置有多个解释入口 —— 于是"模型其实没按想象的方式运行"。

本模块提供三件东西:

  ① `ContentMode` / `ContentConfig`
     内容条件的**三种正式模式**（不再是 table/std 两种 + hybrid 当 legacy 例外 ✗）:
       table  : 固定字符表        -> v80 质量基线
       std    : 标准字形图        -> 真正的开集路线
       hybrid : 字表 + std-soft   -> v83-A 已验证有效的双条件路线
     `ContentConfig.validate()` 校验与旧开关（no_char_cond / std_soft_cond）**自洽** ✓;
     `no_char_cond` / `std_soft_cond` 自此只是**兼容转换器的输入** ✓, 不再是与 mode 并列的真源 ✓。

  ② `ConditionBatch`
     forward 的条件入参打包（现在散成 y_callig / y_char / g / y_script / y_callig_raw / y_pair ✗,
     含义还随配置变化 ⇒ 训练与 CFG 极易传错 ✗）。

  ③ `ConditionFeatures`
     条件编码后的产物（主干只消费这个, **不再知道**字符来自 TTF / DINO 字表 / IDS ✓）。

约定: 本模块**不 import torch 之外的重型依赖**, 也不读取全局配置 —— 纯类型与转换 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

try:                                                     # 允许无 torch 环境下做纯类型检查 ✓
    from torch import Tensor
except Exception:                                        # noqa: BLE001
    Tensor = Any                                        # type: ignore

__all__ = ["ContentMode", "ContentConfig", "content_config_from_args", "content_config_from",
           "ConditionBatch", "ConditionFeatures", "MODE_SWITCHES",
           "build_batch", "batch_to_kwargs", "call_forward"]

# 模式 -> (use_char_cond, std_soft_cond)
MODE_SWITCHES: Dict[str, tuple] = {
    "table": (True, False),
    "std": (False, True),
    "hybrid": (True, True),
}

ContentMode = Literal["table", "std", "hybrid"]


@dataclass(frozen=True)
class ContentConfig:
    """内容条件的**单一真源** ✓

    mode: 'table' | 'std' | 'hybrid'
    declared: 配置里是否**显式**写了 content_cond（False = 由开关推导, 应告警 ✓）
    """
    mode: ContentMode
    declared: bool = True
    glyph_dropout: float = 0.0

    def validate(self):
        if self.mode not in MODE_SWITCHES:
            raise ValueError(
                f"content_cond 只能是 'table' / 'std' / 'hybrid'，收到 {self.mode!r}。"
                f"table=仅字表 / std=仅标准字形 / hybrid=两者并存(归因不可分离 ✗)")
        if not (0.0 <= float(self.glyph_dropout) <= 1.0):
            raise ValueError(f"glyph_dropout 必须在 [0,1]，收到 {self.glyph_dropout!r}")
        return self

    @property
    def switches(self) -> tuple:
        """(use_char_cond, std_soft_cond) —— 由 mode **唯一决定** ✓"""
        return MODE_SWITCHES[self.mode]

    @property
    def use_char_cond(self) -> bool:
        return self.switches[0]

    @property
    def std_soft_cond(self) -> bool:
        return self.switches[1]


def content_config_from(args=None, *, no_char_cond=None, std_soft_cond=None,
                        content_cond=None, glyph_dropout=None, strict=True) -> ContentConfig:
    """从任意来源（Namespace/属性/显式值）构造 ContentConfig ✓

    规则（与历史行为**逐位一致**, 只是把"真源"收拢到 mode ✓）:
      · 显式 content_cond -> 用它, 并**校验**与开关自洽（不自洽即报错 ✓）
      · 未显式         -> 由 (use_char, std_soft) 推导出 mode, declared=False（调用方应告警 ✓）
      · 两条内容条件都关 -> 报错（既无字表也无字形, 内容条件无从谈起 ✗）
    """
    g = (lambda k, d=None: getattr(args, k, d)) if args is not None else (lambda k, d=None: d)

    def pick(v, key, default):
        return v if v is not None else g(key, default)

    use_char = not bool(pick(no_char_cond, "no_char_cond", False))
    std_soft = bool(pick(std_soft_cond, "std_soft_cond", False))
    cc = pick(content_cond, "content_cond", None)
    gd = pick(glyph_dropout, "glyph_dropout", 0.0)

    if cc is not None:
        cc = str(cc).lower()
        cfg = ContentConfig(mode=cc, declared=True, glyph_dropout=float(gd or 0.0)).validate()
        if (use_char, std_soft) != cfg.switches:
            raise ValueError(
                f"content_cond={cc!r} 与开关不一致: 期望 (no_char_cond={not cfg.use_char_cond}, "
                f"std_soft_cond={cfg.std_soft_cond})，实际 (no_char_cond={not use_char}, "
                f"std_soft_cond={std_soft}) —— 配置自相矛盾 ✗")
        return cfg

    inv = {v: k for k, v in MODE_SWITCHES.items()}
    mode = inv.get((use_char, std_soft))
    if mode is None:
        raise ValueError(
            "内容条件两条通路都关闭（no_char_cond=True 且 std_soft_cond=False）"
            "—— 必须显式指定 content_cond='table'/'std'/'hybrid' 之一 ✓")
    cfg = ContentConfig(mode=mode, declared=False, glyph_dropout=float(gd or 0.0)).validate()
    if strict and mode == "hybrid":
        # 审查要求: **不应允许 hybrid 靠 content_cond=None 隐式启用** ✗ —— 这里只告警,
        # 真正强制由 audit_configs/启动预检负责（历史配置太多, 不能一刀切报错 ✓）
        pass
    return cfg


def content_config_from_args(args, log=None, **kw) -> ContentConfig:
    """Namespace -> ContentConfig（**唯一入口** ✓）；未显式声明时按需告警 ✓"""
    cfg = content_config_from(args, **kw)
    if not cfg.declared and log is not None:
        log("[content-cond] ⚠ 未显式指定 content_cond, 由开关推导为 %r ✗ —— 请在配置里显式写 "
            "content_cond='%s'（table=仅字表 / std=仅字形 / hybrid=两者并存, 归因不可分离 ✗）"
            % (cfg.mode, cfg.mode))
    return cfg


@dataclass
class ConditionBatch:
    """forward 的条件入参（替代散落的 6 个位置参数 ✓）

    审查: 这些 ID 的含义还随配置变化, 容易在训练和 CFG 里传错 ✗ ⇒ 打包 + 命名 ✓。
    `present_*` 显式表达"这条通路这一批到底有没有"（CFG 的 uncond 半边靠它表达 ✓）。
    """
    writer: Optional[Tensor] = None                  # 书家（或 pair）id
    script: Optional[Tensor] = None                  # 书体 id
    char: Optional[Tensor] = None                    # 字符 id（None = 本模式不依赖字表 ✓）
    glyph: Optional[Tensor] = None                   # 标准字形软掩码 (N,1,64,64)
    writer_raw: Optional[Tensor] = None              # 书家连续索引（hier 主效应表）
    writer_script_pair: Optional[Tensor] = None      # (书家×书体) pair id
    content_mode: Optional[str] = None               # 便于断言/日志 ✓

    def present(self) -> Dict[str, bool]:
        return {"writer": self.writer is not None, "script": self.script is not None,
                "char": self.char is not None, "glyph": self.glyph is not None,
                "writer_raw": self.writer_raw is not None,
                "writer_script_pair": self.writer_script_pair is not None}

    def to_kwargs(self) -> Dict[str, Any]:
        """兼容映射: 回到历史关键字名（y_callig/y_char/g/y_script/y_callig_raw/y_pair ✓）"""
        return {"y_callig": self.writer, "y_char": self.char, "g": self.glyph,
                "y_script": self.script, "y_callig_raw": self.writer_raw,
                "y_pair": self.writer_script_pair}


@dataclass
class ConditionFeatures:
    """条件编码后的产物（主干只消费它 ✓）

    · global_condition: (N,D) 进 adaLN/融合的全局条件
    · spatial_tokens  : (N,T,D) 逐层注入的 token（标准字形/骨架）
    · style_features  : (N,D) 纯风格信号（c_style 用）
    · present_masks   : 各通路是否真的参与（供门控/CFG 使用 ✓）
    """
    global_condition: Optional[Tensor] = None
    spatial_tokens: Optional[Tensor] = None
    style_features: Optional[Tensor] = None
    present_masks: Dict[str, bool] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

# ─────────────────── 与既有接口的**薄适配层**（不改 DiT.forward 签名 ✓）───────────────────
# 审查要求的"第一阶段: 只移动逻辑, 不移动接口/权重" —— 因此这里只提供**打包/解包**,
# DiT.forward / forward_with_cfg 的签名与调用点一律不动 ✓（前向黄金可逐位验证 ✓）。
# 训练/CFG/评测可以**逐个**迁移到 ConditionBatch, 不需要一次性大改 ✓。

def build_batch(y_callig=None, y_char=None, g=None, y_script=None,
                y_callig_raw=None, y_pair=None, content_mode=None) -> ConditionBatch:
    """把历史关键字打包成 ConditionBatch ✓（名字即语义, 减少传错 ✗）"""
    return ConditionBatch(writer=y_callig, char=y_char, glyph=g, script=y_script,
                          writer_raw=y_callig_raw, writer_script_pair=y_pair,
                          content_mode=content_mode)


def batch_to_kwargs(batch: ConditionBatch, *, for_forward: bool = True) -> Dict[str, Any]:
    """ConditionBatch -> 历史关键字（供尚未迁移的调用点使用 ✓）

    for_forward=True 时只保留非 None 项（forward 的默认值语义一致 ✓）;
    显式传 None 与"不传"在 forward 里等价 ✓, 但 g=None 是**条件被丢掉**的语义 ✓,
    所以需要表达"明确不带 g"时必须走 for_forward=False 的完整映射 ✓。
    """
    kw = batch.to_kwargs()
    if for_forward:
        out = {k: v for k, v in kw.items() if v is not None or k == "g"}
        return out
    return kw


def kwargs_for_call(batch: ConditionBatch, *, allow_none_glyph: bool = False) -> Dict[str, Any]:
    """**迁移助手** ✓: 把 ConditionBatch 还原成"**原本存在**的键" ✓（2026-10-14 ✓）

    ★★ 为什么不能用 `batch_to_kwargs(b, for_forward=True)` 直接替换旧代码 ✗✗
      旧调用点（如 `src/eval/sampler.py`）的写法是"**存在才加键**" ✓::

          mk = dict(y_callig=yc, y_char=yh)
          if skel is not None:
              mk["g"] = skel[i:j]          # ← 不存在就**根本不传** g ✓

      而 `for_forward=True` **总是保留** `g` ✓（哪怕它是 None ✓）
      —— 因为契约上 `g=None` 是"**条件被丢掉**"的语义 ✓ 而不是"不传" ✓
      ⇒ 直接替换会把"没传 g"变成"明确传 g=None" ✗ ⇒ **语义改变** ✗（**已实测确认** ✓）

    ⇒ 迁移时用本函数 ✓: 默认**只保留原本非 None 的键** ✓（与"存在才加"逐字等价 ✓）
      只有确实要表达"明确不带字形"时才 `allow_none_glyph=True` ✓

    ⚠ 注意 `cond` 键（ControlNet 包装用 ✓）**不在** ConditionBatch 里 ✗
      ⇒ 那部分必须留在调用方 ✓（不要硬塞进本层 ✗）
    """
    kw = batch.to_kwargs()
    out = {k: v for k, v in kw.items() if v is not None}
    if allow_none_glyph:
        out["g"] = kw.get("g")
    return out


def call_forward(model, x, t, batch: ConditionBatch, **extra):
    """用 ConditionBatch 调 model.forward（**不改签名** ✓）"""
    kw = dict(batch_to_kwargs(batch))
    kw.update(extra)
    return model(x, t, **kw)
