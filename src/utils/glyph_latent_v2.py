# -*- coding: utf-8 -*-
"""标准字形 latent 字典 v2 查询 (多字体, 字泛化管线).

库: std_glyph_latent_v2/{font}/U+XXXXX.npy (float16 (4,32,32), 与主 latent 同空间)
  font ∈ {kai_gb, kai_st, wei_st, xing_st, li_gb, li_st}  (见 tools/build_std_glyph_latents.py)
  由 8105 通用规范汉字 ∪ mid_clean 数据集字 渲染 256×256 后 sd-vae encode.

用途:
  * zero/few-shot 字符泛化: 训练/推理按 (script, char) 查标准字形 latent 作为
    实例级结构条件 (g), 字体可指定或随机 (字体随机 = 天然的数据增广).
  * 与 v1 (GlyphLatentLookup, kai/li 两库) 的区别: 多字体 + 更全字符覆盖
    (8118 字 vs v1 7923, 且行书有字库了).

用法:
    from src.utils.glyph_latent_v2 import get_glyph_lookup_v2
    lk = get_glyph_lookup_v2()
    g = lk.get(script_id, char)                  # 按 script 默认字体
    g = lk.get(script_id, char, font="kai_st")   # 指定字体
    g = lk.get(script_id, char, random=True)     # 该 script 可用字体里随机
返回 (4,32,32) float32 tensor, 缺失返回 None.
"""
import os
import json
import random as _random

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB_DIR = os.path.join(_HERE, "std_glyph_latent_v2")

# script_id -> 该书体可用字体 (顺序即默认优先级)
SCRIPT_FONTS = {
    0: ["kai_gb", "kai_st", "wei_st"],   # 楷
    3: ["xing_st"],                      # 行
    4: ["li_gb", "li_st"],               # 隶
}
# script_id -> v1 兼容默认 (kai/li)
_SCRIPT_DEFAULT_FONT = {0: "kai_gb", 3: "xing_st", 4: "li_gb"}


class GlyphLatentLookupV2:
    def __init__(self, lib_dir=DEFAULT_LIB_DIR, preload=True):
        self.lib_dir = lib_dir
        self.preload = preload
        self._cache = {}  # font -> {U+XXXXX: float32 tensor (4,32,32)}
        if preload:
            for font in os.listdir(lib_dir):
                d = os.path.join(lib_dir, font)
                if not os.path.isdir(d):
                    continue
                table = {}
                for fn in os.listdir(d):
                    if fn.endswith(".npy"):
                        table[fn[:-4]] = torch.from_numpy(
                            np.load(os.path.join(d, fn)).astype(np.float32))
                self._cache[font] = table

    def fonts_for(self, script_id):
        return SCRIPT_FONTS.get(int(script_id), [])

    def _lookup(self, font, key):
        if self.preload:
            return self._cache.get(font, {}).get(key)
        p = os.path.join(self.lib_dir, font, f"{key}.npy")
        if not os.path.exists(p):
            return None
        return torch.from_numpy(np.load(p).astype(np.float32))

    def get(self, script_id, char, font=None, random=False):
        """返回 (4,32,32) float32 tensor; 缺失返回 None.

        random=True 时在该 script 的可用字体中随机选 (需字体有多个).
        """
        fonts = self.fonts_for(script_id)
        if not fonts:
            return None
        if font is None:
            font = (_random.choice(fonts) if random and len(fonts) > 1
                    else _SCRIPT_DEFAULT_FONT.get(int(script_id), fonts[0]))
        elif font not in fonts:
            return None
        key = f"U+{ord(char):05X}"
        t = self._lookup(font, key)
        if t is None:
            return None
        return t.reshape(4, 32, 32).contiguous()


_glookup_v2 = None


def get_glyph_lookup_v2(lib_dir=DEFAULT_LIB_DIR, preload=True):
    global _glookup_v2
    if _glookup_v2 is None:
        _glookup_v2 = GlyphLatentLookupV2(lib_dir=lib_dir, preload=preload)
    return _glookup_v2
