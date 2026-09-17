# -*- coding: utf-8 -*-
"""评测指标的**唯一实现**。

## 为什么要有这个模块

历史上 `SSIM` 在仓库里被复制了 **14 份**、`skel_iou` **8 份**、`MSE` **4 份**、
`LPIPS` 加载 **6 份**。后果不只是冗余，而是**口径漂移**：
不同文件里的 `_ssim` 用了不同的 `win`（7 / 11）、不同的输入约定
（(H,W,3) float / (B,3,H,W) tensor），算出来的数**不能直接比**，
而报告里却常被并列。见 docs/system/70 §3.2。

## 口径约定（改动这里等于改历史指标，务必谨慎）

| 指标 | 输入 | 约定 |
|---|---|---|
| `mse` | (H,W,C) float **0..1** | 返回 `mean(diff²) * 4.0` —— **乘 4 是历史约定**（等价于 [-1,1] 域下的 MSE），保留以保证与旧报告可比 |
| `ssim` | (H,W,C) float 0..1 | 高斯窗，默认 `win=11, sigma=1.5`；逐通道算再平均 |
| `ssim_torch` | (B,C,H,W) tensor | 同参数的高斯窗版本，用于 GPU 上批算 |
| `skel_iou` | (H,W,C) 或 (N,H,W,C) | 灰度 `mean(axis=-1) < thresh` 二值化后**骨架**的 IoU |

⚠ 旧的 `posters.py:34` 用的是 `win=7` + 均匀窗（box），与这里的 `win=11` 高斯窗
**不是同一个数**。本模块统一为高斯窗版本（= `inference.py` 的原始定义，
也是主评测路径一直在用的），迁移时若发现某处数值变了，先确认它原来用的是哪个。
"""
from __future__ import annotations

import numpy as np

# ── MSE ─────────────────────────────────────────────────────────────────────
MSE_SCALE = 4.0   # 历史约定：等价于在 [-1,1] 域下算 MSE


def mse(pred, gt):
    """(H,W,C) float 0..1 -> float。**乘 4.0 是历史约定**，别改。"""
    return float(np.mean((pred - gt) ** 2)) * MSE_SCALE


# ── SSIM ────────────────────────────────────────────────────────────────────
def _gauss_kernel1d(win, sigma):
    r = win // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def ssim(pred, gt, win=11, data_range=1.0, sigma=1.5, window="gauss"):
    """(H,W,C) float -> float。逐通道算再平均。

    window : ``"gauss"``（默认，主评测路径口径）或 ``"box"``（均匀窗）。
        ⚠ 历史上有两套口径并存且**数值不同**：
          - 主路径（inference / in_mem_eval）: **高斯窗 win=11 sigma=1.5**
          - posters.py / gpu_ablate_eval.py: **均匀窗 win=7**
        本函数把两套都收进来，用 ``window`` 选择 —— **保证数值不变**，
        只是实现只剩一份。默认值 = 主路径口径。
    """
    from scipy.ndimage import correlate1d, uniform_filter
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    if window == "gauss":
        k1d = _gauss_kernel1d(win, sigma)

        def _g(img):
            return correlate1d(correlate1d(img, k1d, axis=0, mode="reflect"),
                               k1d, axis=1, mode="reflect")
    elif window == "box":
        def _g(img):
            return uniform_filter(img, win)
    else:
        raise ValueError(f"window 只支持 'gauss' / 'box'，收到 {window!r}")

    out = []
    for ch in range(pred.shape[2]):
        x = pred[:, :, ch].astype(np.float64)
        y = gt[:, :, ch].astype(np.float64)

        mu_x, mu_y = _g(x), _g(y)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sx2 = _g(x * x) - mu_x2
        sy2 = _g(y * y) - mu_y2
        sxy = _g(x * y) - mu_xy
        m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
        out.append(float(m.mean()))
    return float(np.mean(out))


def ssim_torch(x, y, data_range=1.0, win=11, sigma=1.5):
    """(B,C,H,W) tensor -> (B,) tensor。与 `ssim` 同参数的高斯窗版本（GPU 批算）。

    用 depthwise conv2d 一次算完 batch×channel。
    """
    import torch
    import torch.nn.functional as F
    B, C, H, W = x.shape
    k1d = torch.tensor(_gauss_kernel1d(win, sigma), dtype=x.dtype, device=x.device)
    k2d = (k1d[:, None] * k1d[None, :])[None, None].expand(C, 1, win, win)
    pad = win // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def _g(img):
        # ★ 必须 **reflect** 填充：numpy 版用 `correlate1d(..., mode='reflect')`。
        #   用 `conv2d(padding=pad)`（zero padding）会得到不同的值 ——
        #   实测 64×64 随机图差 0.0032，足够让"同一个模型两个指标"对不上。
        return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="reflect"),
                        k2d, groups=C)

    mu_x, mu_y = _g(x), _g(y)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sx2 = _g(x * x) - mu_x2
    sy2 = _g(y * y) - mu_y2
    sxy = _g(x * y) - mu_xy
    m = ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2))
    return m.mean(dim=(1, 2, 3))


# ── skel IoU ────────────────────────────────────────────────────────────────
def _skeletonize(binary):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary)
    except ImportError:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        skel = np.zeros_like(binary)
        img = binary.copy()
        struct = generate_binary_structure(2, 2)
        while img.any():
            eroded = binary_erosion(img, structure=struct)
            skel |= img & ~eroded
            img = eroded
        return skel


def skel_iou(pred, gt, thresh=0.5):
    """(H,W,C) 或 (N,H,W,C) float -> float。骨架 IoU（不是掩码 IoU）。"""
    if pred.ndim == 3:
        pred, gt = pred[None], gt[None]
    b1 = pred.mean(axis=3) < thresh
    b2 = gt.mean(axis=3) < thresh
    inter = union = 0.0
    for k in range(pred.shape[0]):
        if not b1[k].any() and not b2[k].any():
            inter += 1.0
            union += 1.0
            continue
        if not b1[k].any() or not b2[k].any():
            union += 1.0
            continue
        s1, s2 = _skeletonize(b1[k]), _skeletonize(b2[k])
        inter += float((s1 & s2).sum())
        union += float((s1 | s2).sum())
    return inter / union if union > 0 else 1.0


# ── LPIPS (单例) ─────────────────────────────────────────────────────────────
_LPIPS = {}


def get_lpips(device, net="alex"):
    """进程内单例。返回 (lpips_fn, None) 或 (None, 原因字符串) —— **不吞异常细节**。

    ⚠ 原来多处写法是 `try: ... except Exception: lpips_fn = None`，
    失败只留空列，指标静默缺失（见 docs/system/70 §1.3）。这里把原因返回给调用方，
    由调用方决定是 raise 还是显式降级。
    """
    key = (str(device), net)
    if key in _LPIPS:
        return _LPIPS[key], None
    try:
        import lpips as _lp
        fn = _lp.LPIPS(net=net, verbose=False).to(device).eval()
        for p in fn.parameters():
            p.requires_grad_(False)
        _LPIPS[key] = fn
        return fn, None
    except Exception as e:                                    # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
