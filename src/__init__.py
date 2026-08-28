"""
DiT Calligraphy Package (src)

分层:
  src.model  — 模型 (DiT 主模型 / ControlNet / LoRA)
  src.loss   — 扩散训练目标 (DDPM GaussianDiffusion / FlowMatching) + 辅助 loss
  src.train  — 训练入口 (train.py / train_controlnet.py)
  src.eval   — 推理与评估 (inference.py 核心 + 壳)
  src.utils  — 数据加载 / 采样器 / 字形 latent / 工具
"""

from .model import (
    DiT, DiTBlock, DiT_2Cond, DiT_3Cond,
    DiT_models, DiT_2Cond_models, DiT_3Cond_models, DiT_3Cond_XL_2,
    ControlConditionEncoder, ControlNetDiT, load_main_model,
    inject_lora, upgrade_lora_rank, extract_full_inference,
)
from .utils import (
    MCCDDataset, MCCDLatentDataset, DistributedFactorBalancedSampler,
    GlyphLatentLookup, get_glyph_lookup,
    LatentStructureLoss, LatentStructureProbe,
    find_model,
)

__all__ = [
    "DiT", "DiTBlock", "DiT_2Cond", "DiT_3Cond",
    "DiT_models", "DiT_2Cond_models", "DiT_3Cond_models", "DiT_3Cond_XL_2",
    "ControlConditionEncoder", "ControlNetDiT", "load_main_model",
    "inject_lora", "upgrade_lora_rank", "extract_full_inference",
    "MCCDDataset", "MCCDLatentDataset", "DistributedFactorBalancedSampler",
    "GlyphLatentLookup", "get_glyph_lookup",
    "LatentStructureLoss", "LatentStructureProbe",
    "find_model",
]