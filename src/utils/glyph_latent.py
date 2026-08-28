# -*- coding: utf-8 -*-
"""标准字形 latent 查询(远程): 训练/推理一致地按 (script_id, char) 返回标准字形 latent g。

库: std_glyph_latent/{kai,li}/U+XXXXX.npy  (本地 VAE encode 生成, 已传远程, 4,32,32 float32)
script_id -> 书体: 0=楷->kai, 4=隶->li (KAILI_BOOKS)
返回 g: torch tensor (4,32,32) 与主 latent 同空间; 缺失返回 None(调用方决定是否报错/skip)。
预载入内存(N×4×32×32)以加速训练 getitem。
"""
import os, glob, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(_HERE, "std_glyph_latent")

# script_id -> 书体 key
SCRIPT_TO_BOOK = {0: "kai", 4: "li"}


class GlyphLatentLookup:
    def __init__(self, lib_dir=LIB_DIR, preload=True):
        self.lib_dir = lib_dir
        self.manifest = {}
        mp = os.path.join(lib_dir, "manifest.json")
        if os.path.exists(mp):
            self.manifest = json.load(open(mp, encoding="utf-8"))
        else:
            for book in ["kai", "li"]:
                for p in glob.glob(os.path.join(lib_dir, book, "U+*.npy")):
                    self.manifest[f"{book}/{os.path.splitext(os.path.basename(p))[0]}"] = p
        # 二级索引: book -> {U+XXXXX: (4,32,32) tensor}
        self._cache = {}       # book -> dict[key->tensor]
        self.preload = preload
        if preload:
            self._preload_all()

    def _preload_all(self):
        for book in ["kai", "li"]:
            d = os.path.join(self.lib_dir, book)
            cache_b = {}
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".npy"):
                    key = os.path.splitext(f)[0]
                    cache_b[key] = torch.from_numpy(np.load(os.path.join(d, f)))
            self._cache[book] = cache_b
            print(f"[glyph_latent] preloaded {book}: {len(cache_b)}")

    def key_for(self, script_id, char):
        """script_id(char) -> 'U+XXXXX'"""
        return f"U+{ord(char):05X}"

    def get(self, script_id, char, default_value=0.0):
        """返回 (4,32,32) torch tensor 或 None。default_value 影响缺失时行为。"""
        book = SCRIPT_TO_BOOK.get(int(script_id))
        if book is None:
            return None
        key = self.key_for(script_id, char)
        if self.preload:
            cb = self._cache.get(book)
            if cb is None:
                return None
            t = cb.get(key)
            if t is None:
                return None
            return self._norm(t)
        else:
            p = os.path.join(self.lib_dir, book, f"{key}.npy")
            if not os.path.exists(p):
                return None
            return self._norm(torch.from_numpy(np.load(p)))

    @staticmethod
    def _norm(t):
        """把 latent 统一 reshape 成 (4,32,32); 兼容可能的多余 batch 维。"""
        if t.ndim == 5 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 4 and t.shape[0] == 1:
            t = t[0]
        if t.ndim == 4 or t.ndim == 3:
            return t.reshape(4, 32, 32).contiguous()
        return t.contiguous()


# 全局单例(懒加载), 供 dataset/train 复用避免重复预载
_glookup = None


def get_glyph_lookup():
    global _glookup
    if _glookup is None:
        _glookup = GlyphLatentLookup()
    return _glookup
