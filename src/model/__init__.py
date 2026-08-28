"""src.model — 模型层 (DiT 主模型 + ControlNet + LoRA)。"""

from .dit import (
    TimestepEmbedder, LabelEmbedder, DiTBlock, FinalLayer,
    DiT, DiT_3Cond, DiT_2Cond,
    DiT_models, DiT_2Cond_models, DiT_3Cond_models,
    DiT_3Cond_XL_2,
    DiT_2Cond_XS_2, DiT_2Cond_WS_2, DiT_2Cond_S_2, DiT_2Cond_S_4, DiT_2Cond_S_8,
    DiT_2Cond_B_2, DiT_2Cond_B_4, DiT_2Cond_XL_2,
)
from .controlnet import (
    ControlConditionEncoder, ControlNetDiT, load_main_model,
)
from .lora import (
    inject_lora, upgrade_lora_rank, extract_full_inference,
    extract_lora_and_new_embedders,
)

__all__ = [
    "TimestepEmbedder", "LabelEmbedder", "DiTBlock", "FinalLayer",
    "DiT", "DiT_3Cond", "DiT_2Cond",
    "DiT_models", "DiT_2Cond_models", "DiT_3Cond_models", "DiT_3Cond_XL_2",
    "DiT_2Cond_XS_2", "DiT_2Cond_WS_2", "DiT_2Cond_S_2", "DiT_2Cond_S_4",
    "DiT_2Cond_S_8", "DiT_2Cond_B_2", "DiT_2Cond_B_4", "DiT_2Cond_XL_2",
    "ControlConditionEncoder", "ControlNetDiT", "load_main_model",
    "inject_lora", "upgrade_lora_rank", "extract_full_inference",
    "extract_lora_and_new_embedders",
]