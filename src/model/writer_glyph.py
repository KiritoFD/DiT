"""Writer × glyph residual; the structural token-add path is never replaced."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .components import LowRankSpatialStyleFiLM, get_2d_sincos_pos_embed


class WriterGlyphResidual(nn.Module):
    def __init__(self, width, style_dim, grid, depth, rank=32, at='3,7'):
        super().__init__()
        self.at = tuple(int(s) for s in at.split(','))
        if not self.at or len(set(self.at)) != len(self.at) or any(i < 0 or i >= depth for i in self.at):
            raise ValueError('writer_glyph_at must contain unique valid block indices')
        self.film = LowRankSpatialStyleFiLM(style_dim, width, rank)
        self.film.to_latent = None  # This branch does not supervise or decode skeletons.
        self.projections = nn.ModuleDict({str(i): nn.Linear(width, width, bias=False) for i in self.at})
        self.gains = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(())) for i in self.at})
        for projection in self.projections.values():
            nn.init.normal_(projection.weight, std=.02/math.sqrt(width))
        self.register_buffer('pos', torch.from_numpy(get_2d_sincos_pos_embed(width, grid)).float()[None], persistent=False)

    def residual(self, structural, style, glyph, writer_keep, glyph_keep=None):
        styled = self.film(structural, style, self.pos, cache_styled=False)
        # Ink=1, background=0. A broad 9px neighbourhood allows local restructuring.
        foreground = F.max_pool2d(glyph.float().clamp(0, 1), 9, stride=1, padding=4)
        grid = int(math.isqrt(structural.shape[1]))
        mask = F.adaptive_avg_pool2d(foreground, (grid, grid)).flatten(2).transpose(1, 2)
        mask = mask * writer_keep[:, None, None].to(mask.dtype)
        if glyph_keep is not None:
            mask = mask * glyph_keep.reshape(-1, 1, 1).to(mask.dtype)
        return (styled-structural) * mask.to(structural.dtype)

    def inject(self, x, delta, block):
        key = str(block)
        return x + self.gains[key].to(x.dtype) * self.projections[key](delta)
