# -*- coding: utf-8 -*-
"""style 评价轴（跨字符书家风格度量）。见 `metrics.py` 的模块 docstring。"""
from .metrics import (EmbSet, build_prototypes, cosine, cross_char_retrieval, hwd_cn,
                      l2norm, prototype_margin, style_mmd, style_mmd_by_slot, xctacc)

__all__ = ["EmbSet", "l2norm", "cosine", "cross_char_retrieval", "build_prototypes",
           "prototype_margin", "hwd_cn", "style_mmd", "style_mmd_by_slot", "xctacc"]
