# -*- coding: utf-8 -*-
"""src.loss.structure_mid — 中程结构 aux loss (独立模块, 替代 train.py 内联的 w_std_mid)。

## 为什么单独做 + 为什么不能像旧版那样直接 MSE(x0_pred, 细骨架 g)
旧内联实现: `loss_std_mid = ((x0_pred[mid] - g_lat[mid])**2).mean()`, 其中 g 是 ~3px 的
**细线标准骨架 latent**。但 flow 里 t=0.5 的 `x0_pred = E[x0|x_t]` 天然形态是**软墨迹**:
笔画中心近实墨、边缘一圈灰晕、飞白/毛边等高频被平均掉。拿"硬细线"当 target 去 MSE 一个
"软墨迹", 冲突来自三处: ①宽度(细线 vs 墨迹+晕) ②二值性(硬边 vs 灰度) ③粗细变化(常数 vs 包络)。
模型为了降这个 loss 只能"忽略 g 去记忆 GT", 反而强化记忆 —— 这正是旧 std_mid 效果存疑的根因。

## 本模块的三点修正 (对应 docs 讨论)
1. **载体粗化到中程天然形态**: 不再用细骨架, 而是
   - `blur_gt`: 对 **GT 图像 latent** 做 σ(t) 高斯模糊 (首选, 保留粗细包络+墨色, 与 MMSE 冲突最小);
   - `dilate_skel`: 对骨架做 k(t) 膨胀 (latent 域 max-pool 近似; 精确宽度用 build_std_skel_widths.py 预建);
   σ(t)/k(t) 由 `calibrate_mid_structure.py` 一次前向校准得到, 随 t 联动 (越靠高 t 越糊)。
2. **低通子空间里比**: `LP(z)=interp(avgpool(z,2),2)` (≈pixel 8px 低通) 把"硬边 vs 灰晕"的
   锐度残差吸收掉 -> 膨胀宽度 k 从"致命超参"降级为"不敏感超参", 对 k=5 还是 9 鲁棒。
3. **通道归一化**: `cn(z)` 逐样本逐通道标准化, 去掉墨色/DC/尺度差异, 只剩纯结构(路径+宽度包络)。

loss = mean_{t∈窗口} w(t) · ( LP(cn(x0_pred)) − LP(cn(target)) )²
默认**关闭** (w_std_mid=0), 零破坏; 判据用 ink_ssim / strict, 不是全图 ssim。
"""
import json
import os

import torch
import torch.nn.functional as F


# ── 形态算子 (全部可微, 作用在 (N,C,H,W) latent 上) ─────────────────────────
def lowpass(z, factor=2):
    """≈pixel 8px 低通: avg-pool 下采样再双线性上采样回原尺寸。吃掉灰晕/硬边锐度残差。"""
    if factor <= 1:
        return z
    lp = F.avg_pool2d(z, kernel_size=factor, stride=factor)
    return F.interpolate(lp, size=z.shape[-2:], mode="bilinear", align_corners=False)


def chan_norm(z, eps=1e-5):
    """逐样本逐通道标准化 -> 去掉墨色/DC/尺度, 只留结构。"""
    m = z.mean(dim=(2, 3), keepdim=True)
    s = z.std(dim=(2, 3), keepdim=True)
    return (z - m) / (s + eps)


def _gauss_kernel1d(sigma, device, dtype):
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def blur2d(z, sigma):
    """(N,C,H,W) 各向同性高斯模糊 (depthwise, 不分通道)。sigma<=0 时恒等。"""
    if sigma <= 0:
        return z
    k = _gauss_kernel1d(sigma, z.device, z.dtype)
    C = z.shape[1]
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, k.numel())
    kv = k.view(1, 1, -1, 1).expand(C, 1, k.numel(), 1)
    r = k.numel() // 2
    z = F.conv2d(F.pad(z, (r, r, 0, 0), mode="replicate"), kh, groups=C)
    z = F.conv2d(F.pad(z, (0, 0, r, r), mode="replicate"), kv, groups=C)
    return z


def latent_dilate(z, k):
    """latent 域膨胀近似 (max-pool), k 为 latent 像素半径。精确宽度请用预建多宽度骨架。"""
    if k <= 0:
        return z
    size = 2 * int(round(k)) + 1
    return F.max_pool2d(z, kernel_size=size, stride=1, padding=int(round(k)))


class TSchedule:
    """σ(t)/k(t) 调度: 从校准 json 线性插值; 无 json 时用默认线性 (随 t 增大)。"""

    def __init__(self, grid_t=None, grid_sigma=None, grid_k=None,
                 sigma_slope=4.0, k_slope=2.0, sigma_cap=3.0, k_cap=2.0):
        if grid_t is not None and grid_sigma is not None:
            self.t = torch.tensor(grid_t, dtype=torch.float32)
            self.sigma = torch.tensor(grid_sigma, dtype=torch.float32)
            self.k = torch.tensor(grid_k, dtype=torch.float32) if grid_k is not None else None
            self.interp = True
        else:
            self.interp = False
        self.sigma_slope, self.k_slope = sigma_slope, k_slope
        self.sigma_cap, self.k_cap = sigma_cap, k_cap

    @classmethod
    def from_json(cls, path, **kw):
        if not path or not os.path.exists(path):
            return cls(**kw)
        d = json.load(open(path, encoding="utf-8"))
        return cls(grid_t=d.get("t"), grid_sigma=d.get("sigma"),
                   grid_k=d.get("k"), **kw)

    def _interp_at(self, t, grid_t, grid_v):
        # 线性插值 (端点夹取)
        idx = torch.searchsorted(grid_t, t.contiguous()).clamp(1, grid_t.numel() - 1)
        t0 = grid_t[idx - 1]; t1 = grid_t[idx]
        v0 = grid_v[idx - 1]; v1 = grid_v[idx]
        w = ((t - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0, 1)
        return v0 + w * (v1 - v0)

    def sigma_of(self, t):
        """t: (N,) flow t∈[0,1] -> 每样本 σ (latent px)。"""
        if self.interp:
            return self._interp_at(t, self.t.to(t.device), self.sigma.to(t.device))
        return (self.sigma_slope * t).clamp(0, self.sigma_cap)

    def k_of(self, t):
        if self.interp and self.k is not None:
            return self._interp_at(t, self.t.to(t.device), self.k.to(t.device))
        return (self.k_slope * t).clamp(0, self.k_cap)


class MidStructureLoss:
    """中程结构 aux loss (无参数, 纯函数式)。

    carrier: 'blur_gt' | 'dilate_skel' | 'both'。
    window: 在 a_t=1-t ∈ [alo,ahi] 的中程噪声段生效 (与旧内联口径一致)。
    """

    def __init__(self, carrier="blur_gt", alo=0.35, ahi=0.75, lp_factor=2,
                 sigma_sched=None, w_of_t=None, latent_channels=4):
        self.carrier = carrier
        self.alo, self.ahi = float(alo), float(ahi)
        self.lp_factor = int(lp_factor)
        self.sigma_sched = sigma_sched or TSchedule()
        self.w_of_t = w_of_t          # 可选 callable(t)->(N,) 权重; None=窗口内 1
        self.latent_channels = int(latent_channels)

    def _target(self, gt_lat, skel_lat, t):
        """按载体构造粗化 target (与 pred 同通道数, 无梯度)。"""
        sigma = self.sigma_sched.sigma_of(t)
        k = self.sigma_sched.k_of(t)
        tgt_parts = []
        if self.carrier in ("blur_gt", "both") and gt_lat is not None:
            # 逐样本不同 σ: 按 σ 分桶各模糊一次再按样本取 (σ 值少, 便宜)
            g = gt_lat[:, :self.latent_channels]
            b = _blur_per_sample(g, sigma)
            if self.carrier == "blur_gt":
                return b
            tgt_parts.append(b)
        if self.carrier in ("dilate_skel", "both") and skel_lat is not None:
            s = skel_lat[:, :self.latent_channels]
            d = _dilate_per_sample(s, k)
            if self.carrier == "dilate_skel":
                return d
            tgt_parts.append(d)
        if len(tgt_parts) == 2:
            return 0.5 * (tgt_parts[0] + tgt_parts[1])
        return tgt_parts[0] if tgt_parts else None

    def __call__(self, pred_xstart, gt_lat, skel_lat, t):
        """pred_xstart: (N,C,32,32) 带梯度; gt_lat: GT 图像 latent; skel_lat: 细骨架 latent; t: (N,) flow t。"""
        zero = pred_xstart.sum() * 0.0
        if pred_xstart is None or t is None:
            return zero
        a_t = 1.0 - t.float()                       # 等效 sqrt_alpha (flow)
        active = (a_t >= self.alo) & (a_t <= self.ahi)
        if not bool(active.any()):
            return zero
        tgt = self._target(gt_lat, skel_lat, t)
        if tgt is None:
            return zero
        p = pred_xstart[:, :self.latent_channels].float()
        tgt = tgt.float().detach()
        # 低通子空间 + 通道归一 后比 -> k/σ 不敏感
        dp = lowpass(chan_norm(p), self.lp_factor)
        dt = lowpass(chan_norm(tgt), self.lp_factor)
        se = (dp - dt).pow(2).mean(dim=(1, 2, 3))   # (N,)
        w = active.float()
        if self.w_of_t is not None:
            w = w * self.w_of_t(t.float())
        # sum_over_active / N_total: 与旧 LatentStructureLoss 口径一致, 量级稳定
        return (se * w).sum() / w.sum().clamp_min(1.0)


# ── 逐样本不同 σ/k 的批量近似 (σ 档位少 -> 各档算一次再按样本 gather) ──────────
def _blur_per_sample(z, sigma):
    sig = sigma.detach().cpu().tolist()
    levels = sorted(set(round(s, 2) for s in sig))
    out = torch.zeros_like(z)
    for lv in levels:
        idx = [i for i, s in enumerate(sig) if round(s, 2) == lv]
        sel = torch.tensor(idx, device=z.device)
        out[sel] = blur2d(z[sel], lv)
    return out


def _dilate_per_sample(z, k):
    kv = k.detach().cpu().tolist()
    levels = sorted(set(int(round(x)) for x in kv))
    out = torch.zeros_like(z)
    for lv in levels:
        idx = [i for i, x in enumerate(kv) if int(round(x)) == lv]
        sel = torch.tensor(idx, device=z.device)
        out[sel] = latent_dilate(z[sel], lv)
    return out
