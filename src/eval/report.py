# -*- coding: utf-8 -*-
"""report.py — **离线评分 / 报告层**（2026-10-13 [审查 §4.2]）

审查原文:
  "让生成器完全不知道 GT … 建议定义两个独立的数据对象:
     GenerationSample(sample_id, writer_id, script_id, char_id, glyph_key, seed)
     ScoringSample(sample_id, generated_path, gt_path)
   生成器只接收 GenerationSample，其中**根本没有 gt_path**。只有生成 PNG 落盘之后，
   评分阶段才获得 GT。"
  "算新指标、重新出 poster，不应该重新调用生成模型，也不应该从已经存在的旧图目录
   猜哪个文件是最新结果。"

本模块把这两条做成**结构性保证**（不是约定 ✓）:
  · `GenerationSample` 是 frozen dataclass 且**没有 gt 字段** ✓✓
     => 生成阶段**无法**把 GT 顺手传进去（"输入图当输出图"这类事故从接口上被堵死 ✓）
  · 评分只吃 `ScoringSample`（gen + gt 显式配对）✓
  · `metrics_id` 只覆盖（生成清单哈希 + 指标集合 + 代码版本）✓
    => **重算指标/换 GT 不会改变生成身份** ✓✓；反之生成一变, metrics_id 必变 ✓

刻意保持「零模型依赖」: 本模块**不加载 VAE/DiT/LPIPS**，只读 PNG ✓ —— 这样它可以在
任何机器上离线跑（审查: "不需要重新调用生成模型" ✓）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["GenerationSample", "ScoringSample", "NEAR_BLANK_RATIO",
           "generation_manifest", "generation_id", "metrics_id",
           "write_samples_jsonl", "read_samples_jsonl", "write_metrics",
           "score_samples", "load_rgb01", "ink_mass", "near_blank", "METRIC_SET"]

SCHEMA_VERSION = 1
NEAR_BLANK_RATIO = 0.10          # 与评测口径一致: 墨量比 < 0.10 ⇒ 近白 ✓
METRIC_SET = ("mse", "ssim", "ink_ssim", "ink_iou", "skel_iou", "frag_ratio",
              "hole_pred", "hole_gt", "ink_mass_pred", "ink_mass_gt", "near_blank_pred")


# ───────────────────────────── 两个数据对象（审查给的 ✓）─────────────────────────────
@dataclass(frozen=True)
class GenerationSample:
    """★ **没有 gt_path** ✓✓ —— 生成阶段拿不到 GT（结构上防泄漏 ✓）"""
    sample_id: str
    writer_id: int
    script_id: int
    char_id: Optional[int] = None
    glyph_key: Optional[str] = None
    seed: int = 0

    def validate(self) -> "GenerationSample":
        if not self.sample_id:
            raise ValueError("GenerationSample.sample_id 不能为空 ✗（身份必须稳定 ✓）")
        return self


@dataclass(frozen=True)
class ScoringSample:
    """评分阶段才出现 GT ✓（generated_path + gt_path 显式配对 ✓）"""
    sample_id: str
    generated_path: str
    gt_path: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ScoringSample":
        for k in ("sample_id", "generated_path", "gt_path"):
            if not getattr(self, k):
                raise ValueError(f"ScoringSample.{k} 不能为空 ✗")
        for k in ("generated_path", "gt_path"):
            if not os.path.isfile(getattr(self, k)):
                raise FileNotFoundError(f"ScoringSample.{k} 不存在: {getattr(self, k)!r} ✗ "
                                        f"（缺失就停止, 不猜 ✗）")
        return self


# ───────────────────────────── 生成身份 vs 指标身份 ─────────────────────────────
def generation_manifest(samples: Iterable[GenerationSample]) -> List[Dict[str, Any]]:
    """规范化（排序 ✓）的生成清单 —— 只含**生成**信息, 不含任何 GT ✓"""
    return [asdict(s.validate()) for s in sorted(samples, key=lambda s: s.sample_id)]


def generation_id(samples: Iterable[GenerationSample]) -> str:
    blob = json.dumps(generation_manifest(samples), ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def metrics_id(samples: Iterable[GenerationSample], *, metric_set: Sequence[str] = METRIC_SET,
               code_revision: str = "unknown", gt_digest: str = "") -> str:
    """指标身份 ✓ —— 生成不变 + 指标集不变 ⇒ 可**复用**已有 metrics.json ✓✓

    换 GT（gt_digest 变）或加新指标（metric_set 变）⇒ 新 metrics_id ✓
    但 **generation_id 不变** ✓✓ —— 这正是审查要的"重算不重新生成" ✓。
    """
    blob = json.dumps({"gen": generation_id(samples), "metrics": sorted(metric_set),
                       "code": code_revision, "gt": gt_digest},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_samples_jsonl(path: str, samples: Iterable[GenerationSample]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for s in generation_manifest(samples):
            f.write(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_samples_jsonl(path: str) -> List[GenerationSample]:
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(GenerationSample(**json.loads(line)))
    return out


def write_metrics(eval_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(eval_dir, exist_ok=True)
    p = os.path.join(eval_dir, "metrics.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return p


# ───────────────────────────── 读图与墨量（不含任何模型 ✓）─────────────────────────────
def load_rgb01(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """PNG -> (H,W,1) float32 in [0,1]（灰度 ✓ 与 metrics 的实现口径一致 ✓）"""
    from PIL import Image
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0)[:, :, None]


def ink_mass(a: np.ndarray) -> float:
    """墨量比 = mean(1 - gray) ✓（near_blank 判据用的就是它 ✓）"""
    return float(np.mean(1.0 - a))


def near_blank(a: np.ndarray, ratio: float = NEAR_BLANK_RATIO) -> bool:
    return ink_mass(a) < ratio


# ───────────────────────────── 离线评分（唯一实现 ✓）─────────────────────────────
def score_samples(samples: Sequence[ScoringSample], *, size: Optional[Tuple[int, int]] = None,
                  need_lpips: bool = False, device: str = "cpu",
                  progress: bool = False) -> Dict[str, Any]:
    """逐样本 + 汇总指标 ✓（**不需要模型/GPU** ✓；LPIPS 是可选项 ✓）

    指标全部复用 src/eval/metrics*.py 的**唯一实现** ✓（不另写一套 ✗）。
    """
    from src.eval.metrics import frag_ratio, hole_ratio, mse, ssim, skel_iou
    from src.eval.metrics_ink import ink_iou, ink_ssim

    per: List[Dict[str, Any]] = []
    lpips_fn = None
    if need_lpips:
        from src.eval.metrics import get_lpips
        lpips_fn = get_lpips(device)
    for k, s in enumerate(samples):
        s.validate()
        p, g = load_rgb01(s.generated_path, size), load_rgb01(s.gt_path, size)
        row = {
            "sample_id": s.sample_id, "generated_path": s.generated_path, "gt_path": s.gt_path,
            "mse": float(mse(p, g)), "ssim": float(ssim(p, g)),
            "ink_ssim": float(ink_ssim(p, g)), "ink_iou": float(ink_iou(p, g)),
            "skel_iou": float(skel_iou(p, g)),
            "frag_ratio": float(frag_ratio(p, g)), "hole_pred": float(hole_ratio(p)),
            "hole_gt": float(hole_ratio(g)),
            "ink_mass_pred": ink_mass(p), "ink_mass_gt": ink_mass(g),
            "near_blank_pred": bool(near_blank(p)),
        }
        if lpips_fn is not None:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(np.stack([p, g]).transpose(0, 3, 1, 2)).float()
                row["lpips"] = float(lpips_fn(t[0:1].repeat(1, 3, 1, 1),
                                              t[1:2].repeat(1, 3, 1, 1)).mean().item())
        per.append(row)
        if progress and (k + 1) % 50 == 0:
            print("  [score] %d/%d" % (k + 1, len(samples)), flush=True)

    def _agg(key):
        v = [r[key] for r in per if key in r]
        if not v:
            return None
        a = np.asarray(v, dtype=np.float64)
        return {"mean": float(a.mean()), "p10": float(np.percentile(a, 10)),
                "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}

    agg = {k: _agg(k) for k in METRIC_SET if k not in ("near_blank_pred",)}
    n_blank = sum(1 for r in per if r["near_blank_pred"])
    return {
        "schema_version": SCHEMA_VERSION,
        "n_samples": len(per),
        "metric_set": list(METRIC_SET) + (["lpips"] if need_lpips else []),
        "aggregate": agg,
        "near_blank_rate": (n_blank / len(per)) if per else 0.0,
        "near_blank_threshold": NEAR_BLANK_RATIO,
        "samples": per,
    }
