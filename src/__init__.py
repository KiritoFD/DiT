"""
DiT Calligraphy Package (src)

分层:
  src.model  — 模型 (DiT 主模型 DiT_2Cond / ControlNet)
  src.loss   — 扩散训练目标 (DDPM GaussianDiffusion / FlowMatching) + 辅助 loss
  src.train  — 训练入口 (train.py / train_controlnet.py)
  src.eval   — 推理与评估 (inference.py 核心 + 壳)
  src.utils  — 数据加载 / 采样器 / 字形 latent / 工具

2026-08-31 清理
---------------
删除的死代码：
  - src/model/lora.py（LoRA：inject_lora / upgrade_lora_rank / extract_*）
  - 原版 DiT（单条件，依赖 timm）与 DiT_3Cond（三条件），及二者的 XL 变体
  - DiT_2Cond_XL_2
  - src/eval/{eval_test,eval_full_3cond,backfill_eval}.py（依赖上述 3cond + lora，
    已归档到 _archive/legacy_3cond/，git 历史可查）

当前 pipeline：DiT-2Cond-S/2 预训练 + 1px 骨架 ControlNet。
"""

from .model import (
    DiT_2Cond, DiT_2Cond_models,
    ControlConditionEncoder, ControlNetDiT, load_main_model,
)
from .utils import (
    MCCDDataset, MCCDLatentDataset, DistributedFactorBalancedSampler,
    GlyphLatentLookup, get_glyph_lookup,
    GlyphLatentLookupV2, get_glyph_lookup_v2,
    LatentStructureLoss, LatentStructureProbe,
    find_model,
)

__all__ = [
    "DiT_2Cond", "DiT_2Cond_models",
    "ControlConditionEncoder", "ControlNetDiT", "load_main_model",
    "MCCDDataset", "MCCDLatentDataset", "DistributedFactorBalancedSampler",
    "GlyphLatentLookup", "get_glyph_lookup",
    "GlyphLatentLookupV2", "get_glyph_lookup_v2",
    "LatentStructureLoss", "LatentStructureProbe",
    "find_model",
]
