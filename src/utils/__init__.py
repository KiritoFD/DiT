"""src.utils — 数据与工具层 (数据集、采样器、字形 latent、结构损失辅助)。"""

from .dataset import MCCDDataset
from .latent_dataset import MCCDLatentDataset
from .samplers import DistributedFactorBalancedSampler
from .glyph_latent import GlyphLatentLookup, get_glyph_lookup
from .glyph_latent_v2 import GlyphLatentLookupV2, get_glyph_lookup_v2
from .latent_structure import (
    LatentStructureLoss, LatentStructureProbe,
    downsample_structure, _edge_weighted_gradient_loss, _balanced_bce_dice,
)
from .download import find_model, download_model

__all__ = [
    "MCCDDataset", "MCCDLatentDataset", "DistributedFactorBalancedSampler",
    "GlyphLatentLookup", "get_glyph_lookup",
    "GlyphLatentLookupV2", "get_glyph_lookup_v2",
    "LatentStructureLoss", "LatentStructureProbe",
    "downsample_structure",
    "find_model", "download_model",
]