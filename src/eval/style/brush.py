# -*- coding: utf-8 -*-
"""Brush-Statistics Distance（BSD）—— **不依赖任何神经网络**的可解释笔法距离 ✓。

动机（审查建议 ✓）: 神经风格编码器可能学到的是扫描背景/字面大小/作品老化，而不是笔法 ✗。
BSD 直接从像素统计"用笔"本身，作为独立证据；数值**只在与神经指标对照时才有意义** ✓。

特征族（全部在**书体一致**且**笔画复杂度尽量平衡**的前提下比较 —— 见 `balance_note` ✓）
    笔画宽度      骨架处 2×EDT 的中位数 / P10 / P90 / 变异系数
    方向×宽度     局部方向角分桶 × 宽度分桶的联合直方图
    墨色与边缘    骨架处灰度分布、边缘梯度、飞白比例
    结体比例      墨迹 bbox 长宽比、重心偏移、留白率、墨占比
    弯曲程度      局部方向场的变化率（|∇θ|）在骨架处的分布

距离: 一维分布用 **Wasserstein-1**，联合直方图用 **Jensen–Shannon**，标量用绝对差 ✓。
⚠ 本指标是**自定义辅助指标**，不冒充任何已有书法标准 ✓。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

EPS = 1e-9


@dataclass
class BrushFeat:
    """一张图（或一批聚合后的）笔法特征。所有一维量都是**样本分布**（供 W1 用）✓。"""
    width: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    gray_ink: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    edge: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    curv: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    dir_width: np.ndarray = field(default_factory=lambda: np.zeros((8, 6), np.float32))
    scalars: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.width.size >= 8

    def width_stats(self) -> Dict[str, float]:
        w = self.width
        if w.size == 0:
            return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "cv": float("nan")}
        p10, p50, p90 = np.percentile(w, [10, 50, 90])
        return {"p10": float(p10), "p50": float(p50), "p90": float(p90),
                "cv": float(w.std() / max(w.mean(), EPS))}


def _to_gray01(img: np.ndarray) -> np.ndarray:
    """统一成 (H,W) float32 ∈ [0,1]，**白底黑字**（墨 = 暗）✓。"""
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[-1] == 4:                                   # RGBA: alpha 才是形状 ✓
            a = a[..., :3].astype(np.float32) * (a[..., 3:4].astype(np.float32) / 255.0) + \
                255.0 * (1.0 - a[..., 3:4].astype(np.float32) / 255.0)
            a = a[..., :3]
        a = a.mean(axis=-1)
    a = a.astype(np.float32)
    if a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def features_from_gray(g01: np.ndarray, *, thresh: float = 0.55,
                       n_dir: int = 8, n_width: int = 6) -> BrushFeat:
    """从 (H,W) 灰度图提取笔法特征 ✓。空白图返回 `ok=False` 的特征（不抛异常 ✓）。"""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
    from skimage.morphology import skeletonize

    g = _to_gray01(g01)
    ink = g < thresh
    if ink.sum() < 16:
        return BrushFeat()
    skel = skeletonize(ink)
    if skel.sum() < 8:
        return BrushFeat()
    # ① 笔画宽度 = 骨架处 2×到背景的欧氏距离 ✓
    #    ⚠ 约定: 2×EDT **含中心像素** ⇒ 比几何宽度约大 1 像素（离散偏置 ✓）。
    #      生成与参考用同一约定 ⇒ 分布距离不受影响 ✓。
    edt = distance_transform_edt(ink)
    width = (2.0 * edt[skel]).astype(np.float32)
    # ② 笔画**走向**必须从**骨架几何**取 ✓ —— 笔画内部灰度是平的, 灰度梯度≈0 ✗
    #    （我第一版就错在这里: 采样灰度梯度 ⇒ 方向被端头/边缘主导 ⇒ 横竖笔画分不开 ✗）
    #    做法: 平滑骨架掩码 -> 其梯度**横跨**笔画 -> 转 90° 即得**沿笔画**方向 ✓
    sk_s = gaussian_filter(skel.astype(np.float32), 2.0)
    sgx = sobel(sk_s, axis=1)
    sgy = sobel(sk_s, axis=0)
    ang = (np.arctan2(sgy, sgx) + np.pi / 2.0) % np.pi            # 转 90° => 沿笔画 ✓
    ang_sk = ang[skel]
    gx = sobel(g, axis=1)
    gy = sobel(g, axis=0)
    gray_sk = g[skel]
    # ③ 联合直方图: 方向 × 宽度 ✓（归一化 ⇒ 可比 ✓）
    wbin = np.clip((width / max(np.percentile(width, 95), EPS) * n_width).astype(int), 0, n_width - 1)
    abin = np.clip((ang_sk / np.pi * n_dir).astype(int), 0, n_dir - 1)
    jh = np.zeros((n_dir, n_width), np.float32)
    np.add.at(jh, (abin, wbin), 1.0)
    jh /= max(jh.sum(), EPS)
    # ④ 边缘梯度（墨迹边界处）✓
    mag = np.hypot(gx, gy)
    edge = mag[(mag > 0) & ink]
    # ⑤ 弯曲: 方向场的变化率在骨架处 ✓（圆转 vs 方折）
    curv = np.hypot(sobel(ang, axis=1), sobel(ang, axis=0))[skel]
    # ⑥ 结体比例（归一化到墨迹 bbox ✓）
    ys, xs = np.nonzero(ink)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    bbox = np.zeros_like(ink)
    bbox[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    cy, cx = ys.mean(), xs.mean()
    scalars = {
        "bbox_aspect": float(w / max(h, EPS)),
        "centroid_dy": float((cy - (ys.min() + ys.max()) / 2.0) / max(h, EPS)),
        "centroid_dx": float((cx - (xs.min() + xs.max()) / 2.0) / max(w, EPS)),
        "whitespace": float(1.0 - ink[bbox].mean()),
        "ink_frac": float(ink.mean()),
        # 飞白: 骨架处**偏亮**的比例越高 => 枯笔/飞白越多 ✓
        "flywhite": float((gray_sk > 0.5).mean()),
        "gray_mean": float(gray_sk.mean()),
    }
    return BrushFeat(width=width, gray_ink=gray_sk.astype(np.float32),
                     edge=edge.astype(np.float32), curv=curv.astype(np.float32),
                     dir_width=jh, scalars=scalars)


def features_from_image(path: str, **kw) -> BrushFeat:
    """从图片文件提取（PNG/RGBA 都走 `_to_gray01` 的统一极性 ✓）。"""
    from PIL import Image
    return features_from_gray(np.asarray(Image.open(path)), **kw)


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    """一维 Wasserstein-1（分位数近似，样本量差异大时更稳 ✓）。"""
    if a.size < 4 or b.size < 4:
        return float("nan")
    q = np.linspace(0, 100, 51)
    return float(np.mean(np.abs(np.percentile(a, q) - np.percentile(b, q))))


def bsd(a: BrushFeat, b: BrushFeat) -> Dict[str, float]:
    """Brush-Statistics Distance 各组分量 + 宏平均 ✓（越小越接近该书写家的真实笔法 ✓）。

    ⚠ 任一侧**不可用**（空白/笔画过少）⇒ 全部返回 NaN ✓ —— 绝不用 0 分冒充"完全一致" ✗。
    """
    from scipy.spatial.distance import jensenshannon
    if not (a.ok and b.ok):
        keys = ["width", "gray", "edge", "curv", "dir_width_js", "bsd_macro"]
        return {k: float("nan") for k in keys}
    parts: Dict[str, float] = {
        "width": _w1(a.width, b.width),
        "gray": _w1(a.gray_ink, b.gray_ink),
        "edge": _w1(a.edge, b.edge),
        "curv": _w1(a.curv, b.curv),
        "dir_width_js": float(jensenshannon(a.dir_width.ravel() + EPS,
                                            b.dir_width.ravel() + EPS)),
    }
    for k in sorted(set(a.scalars) & set(b.scalars)):
        if k in ("flywhite", "gray_mean"):
            continue
        parts["sc_" + k] = abs(a.scalars[k] - b.scalars[k])
    vals = [v for v in parts.values() if np.isfinite(v)]
    parts["bsd_macro"] = float(np.mean(vals)) if vals else float("nan")
    return parts


def aggregate(feats: list) -> BrushFeat:
    """把一批字的特征**汇总**成一个（同书家 × 同书体的分布）✓ —— BSD 本就是分布距离 ✓。"""
    fs = [f for f in feats if f.ok]
    if not fs:
        return BrushFeat()
    keys = set(fs[0].scalars)
    for f in fs[1:]:
        keys &= set(f.scalars)
    return BrushFeat(
        width=np.concatenate([f.width for f in fs]),
        gray_ink=np.concatenate([f.gray_ink for f in fs]),
        edge=np.concatenate([f.edge for f in fs]),
        curv=np.concatenate([f.curv for f in fs]),
        dir_width=np.mean([f.dir_width for f in fs], axis=0).astype(np.float32),
        scalars={k: float(np.mean([f.scalars[k] for f in fs])) for k in keys},
    )


balance_note = (
    "⚠ 字形结构本身会影响笔画宽度/方向/留白 ✗ ⇒ 比较两组时必须: "
    "① 书体一致 ✓ ② 尽量按笔画复杂度/部件/结构类别**平衡字符集合** ✓ "
    "（否则横画多的生成字 vs 撇捺多的真迹会给出虚假差异 ✓）"
)
