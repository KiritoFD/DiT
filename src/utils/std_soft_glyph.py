# -*- coding: utf-8 -*-
"""Raw grayscale standard-glyph conditions, independent of the image VAE.

The on-disk store is produced by :mod:`tools.build_std_soft_glyphs` and contains
one unblurred 64x64 uint8 mask per ``(script_id, Unicode character)``.  Ink is
255 and background is 0.  Runtime loading optionally applies Gaussian blur and
returns ``[1,H,W]`` float tensors in ``[0,1]``.

Keeping the base masks unblurred has two useful properties:

* training can sample blur strength without rebuilding the store;
* evaluation uses a fixed sigma while sharing the exact same raster source.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SCRIPT_KEYS = {0: "kai", 3: "xing", 4: "li"}


@dataclass(frozen=True)
class FontFace:
    """Resolved font file and optional face index for TTC collections."""

    path: str
    face: int = 0

    @property
    def label(self) -> str:
        suffix = f"#{self.face}" if self.face else ""
        return f"{os.path.basename(self.path)}{suffix}"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_parts(candidate) -> Tuple[str, int]:
    if isinstance(candidate, str):
        return candidate, 0
    if isinstance(candidate, Mapping):
        return str(candidate["file"]), int(candidate.get("face", 0))
    raise TypeError(f"invalid font candidate: {candidate!r}")


def resolve_font(candidate, font_dirs: Sequence[str]) -> Optional[FontFace]:
    """Resolve a configured font path/name without silently substituting it."""
    name, face = _candidate_parts(candidate)
    expanded = os.path.expandvars(os.path.expanduser(name))
    if os.path.isfile(expanded):
        return FontFace(os.path.abspath(expanded), face)
    target = os.path.basename(name).lower()
    for directory in font_dirs:
        root = os.path.abspath(os.path.expandvars(os.path.expanduser(directory)))
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, os.path.basename(name))
        if os.path.isfile(direct):
            return FontFace(os.path.abspath(direct), face)
        for walk_root, _dirs, files in os.walk(root):
            match = next((f for f in files if f.lower() == target), None)
            if match:
                return FontFace(os.path.join(walk_root, match), face)
    return None


def load_font_config(path: str, extra_font_dirs: Sequence[str] = ()):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    base = os.path.dirname(os.path.abspath(path))
    repo_root = os.path.dirname(base) if os.path.basename(base) == "configs" else os.getcwd()
    dirs: List[str] = []
    for item in list(extra_font_dirs) + list(config.get("font_dirs", [])):
        item = str(item)
        if not os.path.isabs(item) and not item.startswith("~"):
            item = os.path.join(repo_root, item)
        if item not in dirs:
            dirs.append(item)
    resolved = {}
    for sid_text, spec in config["scripts"].items():
        sid = int(sid_text)
        faces = []
        unresolved = []
        for candidate in spec.get("fonts", []):
            face = resolve_font(candidate, dirs)
            if face is None:
                unresolved.append(_candidate_parts(candidate)[0])
            elif face not in faces:
                faces.append(face)
        resolved[sid] = {**spec, "faces": faces, "unresolved": unresolved}
    return config, resolved, dirs


class FontCoverage:
    """Cmap coverage cache shared by audit and rendering."""

    def __init__(self):
        self._cmaps: Dict[FontFace, frozenset] = {}

    def cmap(self, face: FontFace) -> frozenset:
        if face not in self._cmaps:
            from fontTools.ttLib import TTFont

            font = TTFont(face.path, fontNumber=face.face, lazy=True)
            points = set()
            for table in font["cmap"].tables:
                if table.isUnicode():
                    points.update(table.cmap.keys())
            font.close()
            self._cmaps[face] = frozenset(points)
        return self._cmaps[face]

    def supports(self, face: FontFace, char: str) -> bool:
        return len(char) == 1 and ord(char) in self.cmap(face)

    def choose(self, faces: Sequence[FontFace], char: str) -> Optional[FontFace]:
        return next((face for face in faces if self.supports(face, char)), None)


def render_base_mask(char: str, face: FontFace, render_size: int = 512,
                     output_size: int = 64, font_ratio: float = 0.79) -> Image.Image:
    """Render one fixed-em-size glyph and BOX-downsample it without bbox scaling."""
    if len(char) != 1:
        raise ValueError(f"expected one Unicode character, got {char!r}")
    font_px = max(1, int(round(render_size * float(font_ratio))))
    font = ImageFont.truetype(face.path, font_px, index=face.face)
    canvas = Image.new("L", (render_size, render_size), 0)
    draw = ImageDraw.Draw(canvas)
    # Fixed em size; only center the glyph.  We never resize its tight bbox.
    bbox = draw.textbbox((0, 0), char, font=font)
    if bbox is None:
        raise ValueError(f"font {face.label} produced no bbox for U+{ord(char):04X}")
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    origin = (render_size / 2.0 - cx, render_size / 2.0 - cy)
    draw.text(origin, char, font=font, fill=255)
    if np.asarray(canvas, dtype=np.uint8).sum() < 255 * 8:
        raise ValueError(f"font {face.label} rendered an empty glyph for U+{ord(char):04X}")
    return canvas.resize((output_size, output_size), Image.Resampling.BOX)


def soften_mask(image: Image.Image, sigma: float) -> Image.Image:
    sigma = float(sigma)
    image = image.convert("L")
    return image.filter(ImageFilter.GaussianBlur(sigma)) if sigma > 0 else image


def collect_script_characters(csv_paths: Iterable[str], chars: str = "") -> Dict[int, set]:
    result = {sid: set() for sid in SCRIPT_KEYS}
    if chars:
        for sid in result:
            result[sid].update(chars)
    for path in csv_paths:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sid_raw = row.get("script_id", "")
                char = (row.get("character") or "").strip()
                try:
                    sid = int(sid_raw)
                except (TypeError, ValueError):
                    continue
                if sid in result and len(char) == 1:
                    result[sid].add(char)
    return result


class StdSoftGlyphStore:
    """Lookup an audited std-soft store by ``(script_id, character)``."""

    def __init__(self, root: str, *, is_train: bool = False,
                 sigma_min: float = 0.4, sigma_max: float = 1.0,
                 eval_sigma: float = 0.65, preload: bool = True,
                 strict: bool = True, sigma_step: float = 0.05):
        self.root = os.path.abspath(root)
        self.is_train = bool(is_train)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eval_sigma = float(eval_sigma)
        self.strict = bool(strict)
        # ★ 2026-10-11: σ 预离散化。训练时 σ~U(min,max) 是连续随机 —— 但**逐样本现做高斯模糊**
        #   是纯 CPU 开销（288 样本/步 × 8 workers ≈ 2300 次/步），在 CPU 被别的租户抢占的机器上
        #   会把训练拖到 0.6 steps/s（实测 48）。改为**preload 阶段把 σ 网格全部materialize**
        #   成一个 uint8 张量 (K, N, H, W)，用 share_memory_() 让 fork 出来的 worker 共享同一份
        #   （copy-on-write，不额外占内存）→ 取值退化成一次切片 + /255（微秒级）。
        self.sigma_step = max(float(sigma_step), 1e-6)
        index_path = os.path.join(self.root, "index.csv")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"std-soft index not found: {index_path}")
        self._paths: Dict[Tuple[int, str], str] = {}
        self._meta: Dict[Tuple[int, str], dict] = {}
        with open(index_path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["script_id"]), row["character"])
                if key in self._paths:
                    raise ValueError(f"duplicate std-soft key in index: {key!r}")
                path = os.path.abspath(os.path.join(self.root, row["path"]))
                if os.path.commonpath([self.root, path]) != self.root:
                    raise ValueError(f"std-soft path escapes root: {row['path']!r}")
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"std-soft image missing for {key!r}: {path}")
                self._paths[key] = path
                self._meta[key] = row
        self._images: Dict[Tuple[int, str], np.ndarray] = {}
        self._stack: Optional[torch.Tensor] = None      # (K, N, H, W) uint8，σ 网格全量
        self._levels: Dict[float, int] = {}
        self._key_index: Dict[Tuple[int, str], int] = {}
        self._size = 0
        if preload:
            for key, path in self._paths.items():
                self._images[key] = self._read(path)
            self._materialize_sigma_grid()

    @staticmethod
    def _read(path: str) -> np.ndarray:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"std-soft image must be square grayscale, got {array.shape}: {path}")
        return array

    def _sigma_grid(self) -> List[float]:
        """σ 网格（含 sigma_min/max 与 eval_sigma），步长 sigma_step。"""
        lo, hi = self.sigma_min, self.sigma_max
        n = int(round((hi - lo) / self.sigma_step))
        grid = [round(lo + i * self.sigma_step, 6) for i in range(n + 1)]
        if grid[-1] < hi - 1e-9:
            grid.append(round(hi, 6))
        ev = round(float(self.eval_sigma), 6)
        if not any(abs(g - ev) < 1e-9 for g in grid):
            grid.append(ev)
            grid.sort()
        return grid

    def _snap(self, sigma: float) -> float:
        return min(self._levels, key=lambda g: abs(g - float(sigma)))

    def _materialize_sigma_grid(self) -> None:
        """把 σ 网格的全部模糊版本一次性做成 (K,N,H,W) uint8 并共享内存。

        内存：K × N × H × W 字节。K=13（0.40..1.00 步长 0.05）、N=8451、64×64 → **450 MB**；
        `share_memory_()` 后由所有 fork 出来的 dataloader worker 共享（COW），不按 worker 翻倍。
        构建耗时：K×N 次高斯模糊（110k 次 × ~0.1ms ≈ 十秒量级，一次性）。
        """
        keys = list(self._paths)
        self._key_index = {k: i for i, k in enumerate(keys)}
        grid = self._sigma_grid()
        self._levels = {round(s, 6): j for j, s in enumerate(grid)}
        h = self._size or self._read(self._paths[keys[0]]).shape[0]
        self._size = int(h)
        stack = np.zeros((len(grid), len(keys), h, h), dtype=np.uint8)
        for j, s in enumerate(grid):
            for i, k in enumerate(keys):
                base = self._images.get(k)
                if base is None:
                    base = self._read(self._paths[k])
                stack[j, i] = np.asarray(soften_mask(Image.fromarray(base, mode="L"), s),
                                         dtype=np.uint8)
        tensor = torch.from_numpy(stack)
        tensor.share_memory_()
        self._stack = tensor
        print(f"[std-soft] σ 网格已materialize: K={len(grid)} 档 ({grid[0]:.2f}..{grid[-1]:.2f}) × "
              f"N={len(keys)} × {h}×{h} uint8 = {stack.nbytes/2**20:.0f} MB（worker 共享）", flush=True)

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def size(self) -> int:
        if self._size:
            return int(self._size)
        if not self._paths:
            return 0
        first = next(iter(self._paths))
        array = self._images.get(first)
        if array is None:
            array = self._read(self._paths[first])
        return int(array.shape[0])

    def missing(self, pairs: Iterable[Tuple[int, str]]) -> List[Tuple[int, str]]:
        return sorted({(int(sid), str(char)) for sid, char in pairs
                       if (int(sid), str(char)) not in self._paths})

    def assert_coverage(self, pairs: Iterable[Tuple[int, str]], label: str = "dataset"):
        missing = self.missing(pairs)
        if missing and self.strict:
            sample = ", ".join(f"{sid}:{ch}" for sid, ch in missing[:20])
            raise KeyError(f"std-soft store misses {len(missing)} {label} glyphs: {sample}")
        return missing

    def row_index(self, script_id: int, char: str) -> Optional[int]:
        """`(script_id, char)` → σ 网格栈的**行号** ✓（`None` = 缺 ✓）。

        ★ 2026-10-16 [infra_1015 §七] 新增 ✓ —— 供 `MCCDLatentDataset` **批量取数**用 ✓:
          批量路径要一次索引 `self._stack[levels, rows]` ✓ 需要**事先**把每样本的行号
          预算成连续数组 ✓。若直接碰 `_key_index` 会跨类访问私有成员 ✗（不良实践 ✓）
          ⇒ 在此提供**公开**访问器 ✓（单一职责 ✓ 也让 store 的实现细节可自由演进 ✓）
        """
        if self._key_index is None:
            return None
        return self._key_index.get((int(script_id), str(char)))

    def batch_get(self, script_ids, chars, sigmas) -> torch.Tensor:
        """**批量**取 σ 模糊字形 ✓ 返回 `(B, 1, H, W)` float32 ✓。

        ★ 与逐个 `get()` **数值逐位等价** ✓（2026-10-16 [infra_1015 §七] ✓）:
          · σ **由调用方逐样本采好** ✓（★ 文档明确警告: 必须**先采连续 σ**
            再映射到网格 ✓ **不能**直接均匀采样网格索引 ✗ —— 那会改变增强分布 ✗）
          · 走同一条 `_snap` → 网格层 → 同一切片 ✓ 同样的 `.float() / 255` ✓
          · 缺行 ⇒ 与 `get()` 一致: strict 时 `KeyError` ✓ 否则给零 ✓
        ★ 收益 ✓: 旧路径每样本各做一次 `.float().div_(255.0).unsqueeze(0)` ✗
          ⇒ 批量路径**一次**向量化 ✓（B 次小算子 → 1 次 ✓）
        """
        B = len(script_ids)
        if self._stack is None:
            # 慢路径（preload=False ✓ 推理/调试）⇒ 退回逐个 ✓ 语义一致 ✓
            return torch.stack([self.get(int(s), str(c), sigma=float(g))
                                for s, c, g in zip(script_ids, chars, sigmas)], 0)
        rows, levels, miss = [], [], []
        for k in range(B):
            r = self.row_index(script_ids[k], chars[k])
            if r is None:
                if self.strict:
                    raise KeyError(
                        f"std-soft glyph missing: script_id={int(script_ids[k])} "
                        f"char={str(chars[k])!r}")
                miss.append(k)
                rows.append(0)
                levels.append(0)
                continue
            rows.append(r)
            levels.append(self._levels[round(self._snap(float(sigmas[k])), 6)])
        out = self._stack[torch.as_tensor(levels), torch.as_tensor(rows)] \
            .float().div_(255.0).unsqueeze(1)
        if miss:                                    # 缺行 ⇒ 与 get() 一致给零 ✓
            out[torch.as_tensor(miss)] = 0.0
        return out

    def get(self, script_id: int, char: str, sigma: Optional[float] = None) -> torch.Tensor:
        key = (int(script_id), str(char))
        if key not in self._paths:
            if self.strict:
                raise KeyError(f"std-soft glyph missing: script_id={key[0]} char={key[1]!r}")
            return torch.zeros(1, self.size, self.size, dtype=torch.float32)
        if sigma is None:
            sigma = (random.uniform(self.sigma_min, self.sigma_max)
                     if self.is_train else self.eval_sigma)
        # ★ 快路径：preload 阶段已 materialize σ 网格 → 这里只是切片 + /255（微秒级，零 CPU 重活）
        if self._stack is not None:
            j = self._levels[round(self._snap(float(sigma)), 6)]
            i = self._key_index[key]
            return self._stack[j, i].float().div_(255.0).unsqueeze(0)
        # 慢路径（preload=False，推理/调试用）
        array = self._images.get(key)
        if array is None:
            array = self._read(self._paths[key])
        image = soften_mask(Image.fromarray(array, mode="L"), float(sigma))
        out = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(out).unsqueeze(0)


__all__ = [
    "SCRIPT_KEYS", "FontFace", "FontCoverage", "StdSoftGlyphStore",
    "collect_script_characters", "load_font_config", "render_base_mask",
    "resolve_font", "sha256_file", "soften_mask",
]
