# -*- coding: utf-8 -*-
"""现成编码器工厂 —— **不自训** ✓，直接用 timm/transformers 的预训练权重做风格特征。

为什么不自训（用户提醒 ✓）
    自训编码器质量不可控 ✗，会把"编码器本身的问题"混进"生成模型风格好坏"的结论里 ✗。
    改用公开预训练编码器 ⇒ **可复现、可比、无自训引入的偏差** ✓✓；
    再用**有效性判据**挑：
        ① `real_real`（真实真迹→真迹检索）必须**最高** ✓
        ② **XCTAcc > 0.5**（同书家异字 应比 异书家同字 更近 ✓）
        ③ 背景对照（抹掉墨迹）必须**更差** ✓（否则说明特征在吃纸张/扫描 ✗）

⚠ 预处理必须用**每个模型自己的** mean/std/size ✓
    （timm 的 `resolve_data_config` ✓ —— 用错就会得到"看起来有数、其实无意义"的特征 ✗）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# 统一暴露的编码器清单（`--enc` 的取值 ✓）。名称与 timm 一致 ✓。
TIMM_ENCODERS = {
    "dinov2_vits14":        ("vit_small_patch14_dinov2.lvd142m", 518),
    "dinov2_vitb14_reg4":   ("vit_base_patch14_reg4_dinov2.lvd142m", 224),
    # ★ CLIP-S / DINO-S 的**标准 backbone** 是 ViT-L/14 ✓（文献口径 ✓，便于对照已发表数字 ✓）
    "dinov2_vitl14":        ("vit_large_patch14_dinov2.lvd142m", 224),
    "clip_vitl14":          ("vit_large_patch14_clip_224.openai", 224),
    "dino_vits16":          ("vit_small_patch16_224.dino", 224),
    "dino_vitb16":          ("vit_base_patch16_224.dino", 224),
    "mae_vitb16":           ("vit_base_patch16_224.mae", 224),
    "clip_vitb32":          ("vit_base_patch32_clip_224.openai", 224),
    "clip_vitb16_laion2b":  ("vit_base_patch16_clip_224.laion2b", 224),
    "resnet50_a1":          ("resnet50.a1_in1k", 224),
}


@dataclass
class TimmEncoder:
    """timm 预训练编码器的**薄封装**（特征 + 缓存, 与 `StyleEncoder` 接口一致 ✓）。"""
    model: object = None
    device: str = "cpu"
    spec: str = ""
    timm_name: str = ""
    pool: str = "cls"                       # cls | patch_mean | patch_stats
    dim: int = 0
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    size: int = 224
    _cache: dict = field(default_factory=dict)

    @classmethod
    def load(cls, spec: str, *, device: Optional[str] = None, pool: str = "cls",
             verbose: bool = True) -> "TimmEncoder":
        import timm
        import torch
        from timm.data import resolve_data_config

        key = spec.replace("timm:", "")
        if key not in TIMM_ENCODERS:
            raise KeyError("未知编码器 %r ✗；可选: %s" % (spec, sorted(TIMM_ENCODERS)))
        name, native = TIMM_ENCODERS[key]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # ★ 传 img_size 允许把"原生 518"的模型建成 224 输入 ✓
        #   ⚠ 建模尺寸与前处理尺寸**必须一致** ✗ —— 第一版我建 224 但前处理取了
        #     `resolve_data_config` 的 518 ⇒ patch_embed 断言失败 ✓
        #   （`lvd142m` 系列要求 H 严格等于建模尺寸, 不能靠插值 ✓）
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
        m.eval().to(device)
        cfg = resolve_data_config({}, model=m)
        enc = cls(model=m, device=device, spec=spec, timm_name=name, pool=pool,
                  mean=tuple(cfg["mean"]), std=tuple(cfg["std"]),
                  size=224)                       # ★ 与建模尺寸一致 ✓
        if verbose:
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print("[enc] %s (%s) %.1fM | pool=%s | 预处理 mean=%s std=%s size=%d"
                  % (spec, name, n, pool, tuple(round(x, 3) for x in cfg["mean"]),
                     tuple(round(x, 3) for x in cfg["std"]), enc.size))
        enc.dim = enc._probe()
        return enc

    def _probe(self) -> int:
        import torch
        with torch.no_grad():
            return int(self._feats(torch.zeros(1, 3, self.size, self.size,
                                               device=self.device)).shape[1])

    def to_tensor(self, arrs: Sequence[np.ndarray]):
        """按**该模型自己的** mean/std/size 前处理 ✓（白底黑字/RGBA 都统一 ✓）。"""
        import torch
        import torch.nn.functional as F
        out = []
        for a in arrs:
            a = np.asarray(a)
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            if a.shape[-1] == 4:                              # RGBA ⇒ 合成到白底 ✓
                al = a[..., 3:4].astype(np.float32) / 255.0
                a = a[..., :3].astype(np.float32) * al + 255.0 * (1 - al)
            a = a.astype(np.float32)
            if a.max() > 1.0 + 1e-6:
                a = a / 255.0
            out.append(torch.from_numpy(np.clip(a, 0, 1)).permute(2, 0, 1))
        x = torch.stack(out)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", align_corners=False)
        mean = torch.tensor(self.mean).view(1, 3, 1, 1)
        std = torch.tensor(self.std).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous()

    def _feats(self, x):
        import torch
        f = self.model.forward_features(x.to(self.device))
        if f.ndim == 4:                                        # CNN: (B,C,1,1) ✓
            return torch.flatten(f, 1)
        if self.pool == "cls":
            return f[:, 0]
        patches = f[:, 1:] if f.shape[1] > 1 else f
        if self.pool == "patch_mean":
            return patches.mean(1)
        if self.pool == "patch_stats":                         # 均值 + 标准差（纹理/笔法更敏感 ✓）
            return torch.cat([patches.mean(1), patches.std(1)], dim=1)
        raise ValueError("未知 pool=%r ✗" % self.pool)

    @staticmethod
    def mask_ink(a: np.ndarray, thresh: float = 0.55) -> np.ndarray:
        """背景对照: 把墨迹替换为背景色 ⇒ 只留纸张/版式 ✓。"""
        b = np.array(a, copy=True)
        g = np.asarray(b)
        if g.ndim == 3:
            g = g[..., :3].mean(-1)
        g = g.astype(np.float32)
        if g.max() > 1.0 + 1e-6:
            g = g / 255.0
        ink = g < thresh
        if ink.sum() == 0 or not (~ink).any():
            return b
        flat = b[~ink].reshape(-1, b.shape[-1] if b.ndim == 3 else 1)
        bg = np.median(flat, axis=0)
        b[ink] = bg
        return b

    def embed_arrays(self, arrs: Sequence[np.ndarray], *, batch: int = 32,
                     mask_ink: bool = False, progress: bool = False) -> np.ndarray:
        import torch
        arrs = list(arrs)
        if mask_ink:
            arrs = [self.mask_ink(a) for a in arrs]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(arrs), batch):
                f = self._feats(self.to_tensor(arrs[i:i + batch]))
                out.append(f.float().cpu().numpy())
                if progress and (i // batch) % 20 == 0:
                    print("  [embed %s] %d/%d" % (self.spec, i + len(out[-1]), len(arrs)),
                          flush=True)
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)

    # ── 与 StyleEncoder 相同的缓存接口 ✓（CLI 可直接复用 ✓）──
    def cache_key(self, path: str, mask_ink: bool) -> str:
        import hashlib
        st = os.stat(path)
        raw = "|".join([self.spec, self.pool, os.path.abspath(path), str(st.st_size),
                        str(int(st.st_mtime)), str(bool(mask_ink))])
        return hashlib.sha1(raw.encode()).hexdigest()[:20]

    def embed_paths(self, paths: Sequence[str], *, cache: Optional[str] = None,
                    batch: int = 32, mask_ink: bool = False,
                    progress: bool = False) -> Tuple[np.ndarray, List[str]]:
        from PIL import Image
        keys = [self.cache_key(p, mask_ink) for p in paths]
        store: dict = {}
        if cache and os.path.isfile(cache):
            try:
                with np.load(cache, allow_pickle=False) as z:
                    store = {k: z[k] for k in z.files}
            except Exception:                                       # noqa: BLE001
                store = {}
        todo = [i for i, k in enumerate(keys) if k not in store]
        if todo:
            arrs = [np.asarray(Image.open(paths[i]).convert("RGB")) for i in todo]
            embs = self.embed_arrays(arrs, batch=batch, mask_ink=mask_ink, progress=progress)
            for j, i in enumerate(todo):
                store[keys[i]] = embs[j]
        if cache and todo:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            np.savez_compressed(cache, **store)
        return np.stack([store[k] for k in keys]).astype(np.float32), list(keys)
