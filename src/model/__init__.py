"""src.model — 模型层 (DiT 主模型 + ControlNet)。

2026-08-31 清理
---------------
移除的死代码（均已被 DiT_2Cond + ControlNet 取代，且当前 pipeline 不再引用）：
  - ``lora.py`` 整个模块（inject_lora / upgrade_lora_rank / extract_* ）
    当前所有配置均为 ``use_lora: false``；ControlNet 训练是冻结主干 + 训 ctrl 分支，
    不需要 LoRA。
  - 原版 ``DiT``（单条件，依赖 timm）及其组件
  - ``DiT_3Cond``（三条件 callig+script+char）及其所有变体
  - ``DiT_2Cond_XL_2`` / ``DiT_3Cond_XL_2`` 等 XL 变体

保留：
  - ``DiT_2Cond``（当前唯一主干）+ 其变体注册表 ``DiT_2Cond_models``
  - ``ControlConditionEncoder`` / ``ControlNetDiT`` / ``load_main_model``
  - ``modules``（RMSNorm / SwiGLU / 2D-RoPE / QK-Norm 现代化组件）

当前 pipeline 实际只用 **DiT-2Cond-S/2**。
"""

from .dit import (
    TimestepEmbedder, LabelEmbedder,
    DiT_2Cond,
    DiT_2Cond_models,
    DiT_2Cond_XS_2, DiT_2Cond_WS_2, DiT_2Cond_S_2, DiT_2Cond_S_4,
    DiT_2Cond_S_8, DiT_2Cond_B_2, DiT_2Cond_B_4,
)
from .controlnet import (
    ControlConditionEncoder, ControlNetDiT, load_main_model,
)
from . import modules
from .modules import (
    RMSNorm, SwiGLUFeedForward, Attention as ModernAttention,
    DiTBlock as ModernDiTBlock, FinalLayer as ModernFinalLayer,
    PatchEmbed as ModernPatchEmbed,
)

__all__ = [
    "TimestepEmbedder", "LabelEmbedder",
    "DiT_2Cond", "DiT_2Cond_models",
    "DiT_2Cond_XS_2", "DiT_2Cond_WS_2", "DiT_2Cond_S_2", "DiT_2Cond_S_4",
    "DiT_2Cond_S_8", "DiT_2Cond_B_2", "DiT_2Cond_B_4",
    "ControlConditionEncoder", "ControlNetDiT", "load_main_model",
    "modules", "RMSNorm", "SwiGLUFeedForward", "ModernAttention",
    "ModernDiTBlock", "ModernFinalLayer", "ModernPatchEmbed",
]
