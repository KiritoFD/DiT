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
    # S2 (2026-09-22): 三层语义分解 + 局部风格-骨架引导
    StyleHierarchy, ScriptGlyphFiLM, SpatialStyleFiLM, LocalStyleGlyphAdapter,
    DiT_2Cond,
    DiT_2Cond_models,
    DiT_2Cond_XS_2, DiT_2Cond_WS_2, DiT_2Cond_S_2, DiT_2Cond_S_4,
    DiT_2Cond_S_8, DiT_2Cond_B_2, DiT_2Cond_B_4,
)
# ── ControlNet 线已归档（2026-09-17）────────────────────────────────────────
# 原: `from .controlnet import ControlConditionEncoder, ControlNetDiT, load_main_model`
# ControlNet 双臂方案已废弃，实现移到 `src/model/legacy/controlnet.py`。
# 这里**不再默认导入** —— 它会把整条废弃依赖链拉进每次 `import src.model`。
# 仍需要它的地方（如 cpu_eval_worker 的 ctrl_pair 模式）请显式:
#     from src.model.legacy.controlnet import ControlNetDiT
try:                                   # 兼容: 旧 ckpt/脚本若还指望顶层可见
    from .legacy.controlnet import (   # noqa: F401
        ControlConditionEncoder, ControlNetDiT, load_main_model,
    )
except Exception:                      # noqa: BLE001
    ControlConditionEncoder = ControlNetDiT = load_main_model = None
from . import modules
from .modules import (
    RMSNorm, SwiGLUFeedForward, Attention as ModernAttention,
    DiTBlock as ModernDiTBlock, FinalLayer as ModernFinalLayer,
    PatchEmbed as ModernPatchEmbed,
)

__all__ = [
    "TimestepEmbedder", "LabelEmbedder",
    "StyleHierarchy", "ScriptGlyphFiLM", "SpatialStyleFiLM", "LocalStyleGlyphAdapter",
    "DiT_2Cond", "DiT_2Cond_models",
    "DiT_2Cond_XS_2", "DiT_2Cond_WS_2", "DiT_2Cond_S_2", "DiT_2Cond_S_4",
    "DiT_2Cond_S_8", "DiT_2Cond_B_2", "DiT_2Cond_B_4",
    "ControlConditionEncoder", "ControlNetDiT", "load_main_model",
    "modules", "RMSNorm", "SwiGLUFeedForward", "ModernAttention",
    "ModernDiTBlock", "ModernFinalLayer", "ModernPatchEmbed",
]
