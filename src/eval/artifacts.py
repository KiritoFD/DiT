# -*- coding: utf-8 -*-
"""artifacts.py — **ArtifactWriter**（2026-10-13 [审查 §4.1/§4.2]）

审查链条里这一环的职责: "ArtifactWriter（保存生成 PNG + manifest）" ✓

关键设计（对齐审查"不应该从旧图目录猜哪个文件是最新结果" ✗）:
  · 图与**生成清单**在同一处落盘 ✓✓ —— `samples.jsonl` 是**生成的副产品**,
    不是事后从目录里猜出来的 ✗ ⇒ 任何离线评分都能拿到权威的生成身份 ✓
  · 文件名由 `sample_id` **确定性**映射（sanitize 后 + 短哈希 ✓）⇒ 不会撞名, 也不依赖顺序 ✓
  · **原子写**（临时文件 + os.replace ✓）—— 半写入的 PNG 不会被后续评分当成成品 ✗
  · 本模块**不知道 GT** ✓✓（没有 gt 参数、没有 gt 字段 ✓）—— 与 report.py 的 GenerationSample 一致 ✓

与 report.py 的分工:
    artifacts.py  → 生成侧（写图 + 记身份）;  report.py → 评分侧（读图 + 读 GT + 算指标）
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.report import GenerationSample, generation_manifest, read_samples_jsonl

__all__ = ["SAMPLES_JSONL", "safe_name", "sample_relpath", "iter_generated",
           "next_index", "write_png", "GenerationWriter", "read_manifest"]

SAMPLES_JSONL = "samples.jsonl"
_SAFE = re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+")


def safe_name(sample_id: str, maxlen: int = 48) -> str:
    """sample_id -> 安全文件名（保留中英文与数字 ✓；**始终**附加短哈希 ✓）。

    ★★ 2026-10-16 [新版审查 §三 P1 ✓]: 旧实现**只在过长时**才加哈希 ✗
      ⇒ `"a/b"` 与 `"a b"` 清洗后**都是** `"a_b"` ✗✗ ⇒ **文件名碰撞** ✗
        后写的 PNG 覆盖先写的 ✗（审查已做最小复现 ✓）
    ⇒ 修法（审查建议 ✓ "sample_id 文件名**始终附加**原始 ID 的稳定哈希，
       而不是仅在超长时附加" ✓）: **无条件**追加 8 位 sha256 ✓
      ⇒ 任意两个不同 sample_id ⇒ 文件名必不同 ✓✓
    ★ 副作用（如实 ✓）: 文件名格式变化 ✗ —— 本模块当前 `src` 内无调用方 ✓
      （新版审查 §六 ✓）⇒ 无存量文件受影响 ✓ 接入前修正正是时机 ✓✓
    """
    s = _SAFE.sub("_", str(sample_id)).strip("_") or "s"
    h = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8]
    if len(s) > maxlen:
        s = s[:maxlen]
    return f"{s}_{h}"


def sample_relpath(sample_id: str, subdir: str = "samples", ext: str = "png") -> str:
    """确定性相对路径 ✓（同 id 必得同路径 ✓；不依赖生成顺序 ✓）"""
    return os.path.join(subdir, "%s.%s" % (safe_name(sample_id), ext))


def write_png(path: str, arr: np.ndarray) -> str:
    """原子写 PNG ✓（arr: (H,W) 或 (H,W,C), float [0,1] 或 uint8 ✓）"""
    from PIL import Image
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # NOTE 2026-10-13: 必须显式 format="PNG" —— PIL 按**扩展名**推格式, 对 .tmp 会抛
    #   ValueError: unknown file extension: .tmp（实测踩到 ✗）
    Image.fromarray(a).save(tmp, format="PNG")
    os.replace(tmp, path)                      # 原子替换 ✓（半写入不会被看到 ✓）
    return path


def read_manifest(eval_dir: str) -> List[GenerationSample]:
    """读回生成清单（缺失返回空表 ✓ —— 由调用方决定是否报错）"""
    p = os.path.join(eval_dir, SAMPLES_JSONL)
    return read_samples_jsonl(p) if os.path.isfile(p) else []


def iter_generated(eval_dir: str, subdir: str = "samples") -> Iterator[Tuple[str, str]]:
    """遍历已落盘的生成图: (sample_id, path) ✓（按文件名排序 ⇒ 确定 ✓）"""
    d = os.path.join(eval_dir, subdir)
    if not os.path.isdir(d):
        return
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".png"):
            yield fn[:-4], os.path.join(d, fn)


def next_index(eval_dir: str, subdir: str = "samples") -> int:
    """已有多少张（供"续跑"时对账 ✓）"""
    return sum(1 for _ in iter_generated(eval_dir, subdir))


class GenerationWriter:
    """边生成边落盘 —— **图与身份同源** ✓✓

    用法:
        w = GenerationWriter(eval_dir)
        for cond in conds:                       # cond 只含生成所需信息 ✓（无 GT ✓）
            img = sample(...)
            w.add(GenerationSample(sample_id=..., ...), img)
        w.close()                                 # 写 samples.jsonl（含 generation_id ✓）
    """

    def __init__(self, eval_dir: str, subdir: str = "samples",
                 extra: Optional[Dict[str, Any]] = None):
        self.eval_dir = eval_dir
        self.subdir = subdir
        self.extra = dict(extra or {})
        self._samples: List[GenerationSample] = []
        self._seen = set()
        os.makedirs(os.path.join(eval_dir, subdir), exist_ok=True)

    def add(self, sample: GenerationSample, image: np.ndarray) -> str:
        s = sample.validate()
        if s.sample_id in self._seen:
            raise ValueError("sample_id 重复: %r ✗（身份必须唯一 ✓）" % s.sample_id)
        self._seen.add(s.sample_id)
        p = os.path.join(self.eval_dir, sample_relpath(s.sample_id, self.subdir))
        write_png(p, image)
        self._samples.append(s)
        return p

    def add_many(self, samples: Sequence[GenerationSample],
                 images: Iterable[np.ndarray]) -> List[str]:
        return [self.add(s, im) for s, im in zip(samples, images)]

    @property
    def n(self) -> int:
        return len(self._samples)

    def close(self) -> Dict[str, Any]:
        """写 samples.jsonl（**只含身份, 不含 GT/路径** ✓✓）+ 一个可核对的 summary ✓"""
        from src.eval.report import generation_id, write_samples_jsonl
        p = os.path.join(self.eval_dir, SAMPLES_JSONL)
        write_samples_jsonl(p, self._samples)
        gid = generation_id(self._samples) if self._samples else ""
        summary = {"n_samples": self.n, "generation_id": gid, "samples_jsonl": p}
        summary.update(self.extra)
        with io.open(os.path.join(self.eval_dir, "generation.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        return summary

    def manifest(self) -> List[Dict[str, Any]]:
        return generation_manifest(self._samples)


if __name__ == "__main__":      # 自检（无副作用 ✓）
    import tempfile
    d = tempfile.mkdtemp()
    w = GenerationWriter(d)
    w.add(GenerationSample("写法/测试 01", 1, 0, 2016, "u+6236", 7), np.zeros((8, 8), np.float32))
    w.add(GenerationSample("b", 2, 1, None, None, 8), np.ones((8, 8), np.float32))
    info = w.close()
    print("  ✓ 自检:", info["n_samples"], "张 |", os.path.basename(info["samples_jsonl"]))
    print("  ✓ 清单不含 GT/图片路径:", "gt_path" not in io.open(info["samples_jsonl"], encoding="utf-8").read())
    print("  ✓ 读回一致:", len(read_manifest(d)) == 2, "| iter:", sorted(k for k, _ in iter_generated(d)))
