# -*- coding: utf-8 -*-
"""Writer Style Encoder 封装 —— **直接复用已有分类器**，无需先训练新模型 ✓。

用哪个表示（`kind`）
    "callig"（默认）: `callig_head` 的**倒数第二层** 512D —— 专为 2243 位书家训练 ✓（风格性最强）
    "cls"          : dinov2 主干 CLS 384D —— 通用视觉特征（`extract_features` 的返回 ✓）
    ⚠ 两者都**与 DiT 完全无关**（独立训练的判别器 ✓），不构成"循环论证" ✗。

预处理（必须与判别器训练时一致，否则风格特征不可信 ✗）
    RGB float ∈ [0,1] -> **bicubic resize 224×224** -> **ImageNet mean/std 归一化** ✓
    ⇒ 本模块**自带口径自证**: `validate_accuracy()` 用真实数据测 calligrapher Top-1,
      应接近 ckpt 里记录的 `val_callig`（≈69.5%）✓；差太多说明预处理选错了 ✗。

背景对照（审查要求 ✓）
    `mask_ink=True` 时把**墨迹**替换成背景色, 只留纸张/版式 ⇒ 若识别率仍高, 说明编码器
    依赖的是碑帖材质/扫描来源, **不是笔法** ✗ —— 这是必做的前置检验之一 ✓。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE = 224


@dataclass
class StyleEncoder:
    """风格编码器（薄封装 torch 模型；`embed_*` 返回 L2 归一化前的原始特征 ✓）。"""
    net: object = None                      # MultiTaskCalligraphyEvalNet
    device: str = "cpu"
    kind: str = "callig"
    ckpt: str = ""
    dim: int = 512
    _feat: list = field(default_factory=list)

    # ── 构造 ──
    @classmethod
    def load(cls, ckpt: str = "pretrained_models/eval_classifier.pt", *,
             device: Optional[str] = None, kind: str = "callig",
             num_characters: int = 7765, num_calligraphers: int = 2243,
             num_scripts: int = 12, verbose: bool = True) -> "StyleEncoder":
        import torch

        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(ckpt) and not os.path.exists(ckpt):
            ckpt = os.path.join(here, ckpt)
        assert os.path.isfile(ckpt), f"找不到分类器权重: {ckpt} ✗"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        import sys
        ev = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        if os.path.abspath(ev) not in [os.path.abspath(p) for p in sys.path]:
            sys.path.insert(0, os.path.abspath(ev))
        from eval_models import MultiTaskCalligraphyEvalNet    # noqa: E402

        net = MultiTaskCalligraphyEvalNet(num_characters, num_calligraphers, num_scripts,
                                         backbone="dinov2_vits14", pretrained=False)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        r = net.load_state_dict(sd, strict=False)
        if verbose:
            print(f"[style-enc] {os.path.basename(ckpt)}: missing={len(r.missing_keys)} "
                  f"unexpected={len(r.unexpected_keys)}"
                  + (f" | ckpt val_callig={ck.get('val_callig')}" if isinstance(ck, dict) else ""))
        assert not r.missing_keys and not r.unexpected_keys, "权重未完全匹配 ✗（口径可疑 ✓）"
        net.to(device).eval()
        enc = cls(net=net, device=device, kind=kind, ckpt=ckpt)
        # ★ 必须先**注册 hook** 再探维度 ✗（我第一版顺序反了 ⇒ hook 未触发 ✓）
        #   callig_head = [Linear, BN, ReLU, Dropout, Linear] ⇒ 索引 2 = 512D 倒数第二层 ✓
        if kind == "callig":
            def _grab(mod, inp, out):          # ★ hook **必须返回 None** ✓
                enc._feat.clear()              #   （返回值会被当成该层的**新输出** ✗
                enc._feat.append(out)          #    我第一版返回 tuple ⇒ 下游 Dropout 报 TypeError ✓）
                return None
            net.callig_head[2].register_forward_hook(_grab)
        enc.dim = enc._probe_dim()
        return enc

    def _probe_dim(self) -> int:
        import torch
        with torch.no_grad():
            x = torch.zeros(2, 3, SIZE, SIZE, device=self.device)
            return int(self._feats(x).shape[1])

    # ── 前处理 ──
    @staticmethod
    def to_tensor(arrs: Sequence[np.ndarray]):
        """RGB uint8/float [0,1] 数组序列 -> 归一化后的 (B,3,224,224) 张量 ✓。"""
        import torch
        import torch.nn.functional as F
        out = []
        for a in arrs:
            a = np.asarray(a)
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            if a.shape[-1] == 4:                       # RGBA ⇒ 合成到白底 ✓
                al = a[..., 3:4].astype(np.float32) / 255.0
                a = a[..., :3].astype(np.float32) * al + 255.0 * (1 - al)
            a = a.astype(np.float32)
            if a.max() > 1.0 + 1e-6:
                a = a / 255.0
            out.append(torch.from_numpy(np.clip(a, 0, 1)).permute(2, 0, 1))
        x = torch.stack(out)
        if x.shape[-1] != SIZE or x.shape[-2] != SIZE:
            x = F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous()

    @staticmethod
    def mask_ink(a: np.ndarray, thresh: float = 0.55) -> np.ndarray:
        """把墨迹替换为**背景色**（只留纸张/版式）⇒ 背景对照 ✓。"""
        b = np.array(a, copy=True)
        g = np.asarray(b)
        if g.ndim == 3:
            g = g[..., :3].mean(-1)
        g = g.astype(np.float32)
        if g.max() > 1.0 + 1e-6:
            g = g / 255.0
        ink = g < thresh
        if ink.sum() == 0:
            return b
        bg = np.median(b[~ink].reshape(-1, b.shape[-1] if b.ndim == 3 else 1), axis=0) \
            if (~ink).any() else 255
        b[ink] = bg
        return b

    # ── 特征 ──
    def _feats(self, x):
        if self.kind == "callig":
            self._feat.clear()
            self.net(x.to(self.device))
            assert self._feat, "hook 未触发 ✗"
            return self._feat[0]
        return self.net.extract_features(x.to(self.device))

    def embed_arrays(self, arrs: Sequence[np.ndarray], *, batch: int = 32,
                     mask_ink: bool = False, progress: bool = False) -> np.ndarray:
        """批量提特征 ✓（返回 (N,d) float32，未归一化；调用方用 `l2norm` ✓）。"""
        import torch
        arrs = list(arrs)
        if mask_ink:
            arrs = [self.mask_ink(a) for a in arrs]
        out: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(arrs), batch):
                x = self.to_tensor(arrs[i:i + batch])
                f = self._feats(x)
                out.append(f.float().cpu().numpy())
                if progress and (i // batch) % 10 == 0:
                    print(f"  [embed] {i + len(x)}/{len(arrs)}", flush=True)
        return np.concatenate(out, 0) if out else np.zeros((0, self.dim), np.float32)

    # ── 自证 & 缓存 ──
    def validate_accuracy(self, samples: Sequence[Tuple[str, int]], *, mask_ink: bool = False,
                          batch: int = 32) -> Dict[str, float]:
        """用真实样本测 calligrapher Top-1 ✓ —— 应与 ckpt 的 `val_callig` 接近（口径自证 ✓）。"""
        import torch
        from PIL import Image
        ys, pred = [], []
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            arrs = [np.asarray(Image.open(p).convert("RGB")) for p, _ in chunk]
            if mask_ink:
                arrs = [self.mask_ink(a) for a in arrs]
            with torch.no_grad():
                x = self.to_tensor(arrs)
                _, lg, _ = self.net(x.to(self.device))
            pred += lg.argmax(-1).cpu().tolist()
            ys += [int(y) for _, y in chunk]
        p = np.array(pred)
        y = np.array(ys)
        return {"n": int(len(y)), "top1": float((p == y).mean() * 100.0) if len(y) else float("nan")}

    def cache_key(self, path: str, mask_ink: bool) -> str:
        st = os.stat(path)
        raw = "|".join([os.path.abspath(self.ckpt), self.kind, os.path.abspath(path),
                        str(st.st_size), str(int(st.st_mtime)), str(bool(mask_ink))])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def embed_paths(self, paths: Sequence[str], *, cache: Optional[str] = None,
                    batch: int = 32, mask_ink: bool = False,
                    progress: bool = False) -> Tuple[np.ndarray, List[str]]:
        """按路径提特征，带 npz 缓存 ✓（重复评测时省时；缓存键含 mtime/size ✓）。"""
        from PIL import Image
        keys = [self.cache_key(p, mask_ink) for p in paths]
        store: Dict[str, np.ndarray] = {}
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
        return (np.stack([store[k] for k in keys]).astype(np.float32), list(keys))
