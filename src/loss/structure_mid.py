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
    """σ(t)/k(t) 调度: 从校准 json 线性插值; 无 json 时用默认线性 (随 t 增大)。

    ⚠ σ/k 的**单位与语义随载体通路而变**, 两条通路不能共用一套默认值:
      · **latent 域** (`_target`, 走 blur2d / latent_dilate):
        k 是 max-pool 半径, 单位 latent cell, 量级 0~2。默认 slope=2.0 / cap=2.0。
      · **pixel 域** (`_target_png`, 从预建 PNG 选档):
        k 用来在 {3,5,7,9,11}px 五档里挑最近档, 换算后 k_px=k*8 要落在 1~5。
        若沿用 slope=2.0 -> t=0.35 就 k_px=5.6 冲顶, **前 4 档永不可达**,
        多宽度预建形同虚设。故 png 通路用 slope=0.8 / cap=1.0
        (t=0.75 才到 k_px=4.8≈最大档, 窗口内五档都能出现)。
    """

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
    def for_png(cls, grid_t=None, grid_sigma=None, grid_k=None,
                sigma_slope=0.5, k_slope=0.8, sigma_cap=0.55, k_cap=1.0):
        """png 通路默认调度 —— 刻度**对齐预建档位**, 保证窗口内各档都能被选到。

        预建档位 (pixel 域, 见 tools/prepare_mid_carriers.py):
          w  ∈ {3,5,7,9,11}          -> pixel 半径 (w-1)/2 ∈ {1,2,3,4,5}
          σ  ∈ {0.5,1,1.5,2,2.5,3,4} -> pixel σ
        latent 域换算 (downscale=8):
          k_lat  ∈ {0.125,0.25,0.375,0.5,0.625}
          σ_lat  ∈ {0.0625,0.125,0.1875,0.25,0.3125,0.375,0.5}
        故:
          k_slope=0.8 / k_cap=1.0     -> t=0.75 时 k_lat=0.6 ≈ 最大档
          σ_slope=0.5 / σ_cap=0.55    -> t=0.75 时 σ_lat=0.375 ≈ 最大档
        ★ 沿用 latent 域默认 (σ_slope=4/cap=3, k_slope=2/cap=2) 会瞬间冲顶,
          只剩最大档可选, 预建的多档位**静默作废**。
        """
        return cls(grid_t=grid_t, grid_sigma=grid_sigma, grid_k=grid_k,
                   sigma_slope=sigma_slope, k_slope=k_slope,
                   sigma_cap=sigma_cap, k_cap=k_cap)

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

    ── ★ png_carriers 模式 (纯 CPU 预建, 不占 GPU / 不跑 VAE encode) ──────────
    本类有两条 target 来源:
      (a) **latent 域实时算子** (原路径): 传 gt_lat / skel_lat, 内部 blur2d / latent_dilate。
      (b) **pixel 域预建 PNG** (新路径): 传 png，形状 (N,K,256,256) uint8, 由
          tools/prepare_mid_carriers.py 预先算好, 训练时只做 8x 下采样对齐 latent 网格。
    路径 (b) 的意义: 训练时**不需要** VAE encode 这些载体 (省 GPU), 且
    skeletonize / 精确宽度膨胀只在 CPU 侧算一次 (latent_dilate 的 max-pool 只是近似)。
    几何: PNG 256x256 -> latent 32x32, 恰好 8x。一个 8x8 像素块对应一个 latent cell。
    下采样用 **area mean** (与 VAE encoder 的首层卷积感受野同量级), 再按需 min/max:
      - skel (白底黑线) 用 **min**: 块内只要有一像素是墨就认墨 -> 保住细线,
        等价于"先腐蚀背景"; 用 mean 会把 1px 细线摊薄成灰 -> 与骨架语义不符。
      - blur_gt 已是连续灰度, 用 **mean**。
    """

    def __init__(self, carrier="blur_gt", alo=0.35, ahi=0.75, lp_factor=2,
                 sigma_sched=None, w_of_t=None, latent_channels=4,
                 png_keys=None, png_downscale=8):
        self.carrier = carrier
        self.alo, self.ahi = float(alo), float(ahi)
        self.lp_factor = int(lp_factor)
        # png 通路: png_keys 是 dataset.mid_keys 的子序列, 决定从 (N,K,H,W) 里选哪几个
        self.png_keys = list(png_keys) if png_keys else None
        self.png_downscale = int(png_downscale)
        # ★ 两条通路的 σ/k 默认刻度不同 (见 TSchedule docstring) —— 给 png_keys
        #   却沿用 latent 域的 k_slope=2.0 会让 k 早早饱和在最大档。
        #   未显式给 sigma_sched 时, 按通路自动选合适默认。
        if sigma_sched is None:
            sigma_sched = (TSchedule.for_png() if self.png_keys else TSchedule())
        self.sigma_sched = sigma_sched
        self.w_of_t = w_of_t          # 可选 callable(t)->(N,) 权重; None=窗口内 1
        self.latent_channels = int(latent_channels)

    # ── png -> latent 网格 (无参数, 不跑 VAE) ────────────────────────────────
    def _png_to_grid(self, png_u8, sel=None):
        """(N,K,256,256) uint8 -> (N,|sel|,32,32) float, 归一到 [0,1] 灰度。

        sel: 要处理的 column 下标 (None=全部)。**只算需要的列** —— 传 sel 后
             中间张量按 |sel|/K 缩小 (12 档里 skel 只要 5 档, blur 只要 7 档)。

        ★ **只算 area mean**, 不算 min-pool:
          (a) 语义: min-pool 只记"块内有没有墨"(presence), **丢掉粗细信息** ——
              实测 2px 与 10px 竖线在 8x 下采样后 min-pool 都是同一批格子。
              而 area mean 的灰度 ∝ 块内墨占比 ∝ 局部线宽, 正好保住多宽度的意义。
          (b) 显存: 全量算 avg+min 两份会让中间张量翻倍 (batch=240 实测 1675 MiB);
              只留 avg 且只算需要的列 -> 921 MiB -> 更少。
        """
        f = self.png_downscale
        if sel is not None:
            # 用 index_select 而不是切片: sel 是"类 3,4,5.."这类不连续下标与否都适用
            png_u8 = png_u8.index_select(1, torch.as_tensor(
                sel, device=png_u8.device, dtype=torch.long))
        n, k, h, w = png_u8.shape
        # ★ 先 reshape 再转 float: 比先转 float 再 reshape 少一份大张量
        x = png_u8.reshape(n * k, 1, h, w).float()
        x.div_(255.0)                              # 白底=1, 墨=0 (PNG 是 L 模式)
        avg = F.avg_pool2d(x, kernel_size=f, stride=f)
        return avg.reshape(n, k, *avg.shape[-2:])

    def _target_png(self, png, t):
        """从预建 PNG 取载体 target。png: (N,K,256,256) uint8; 返回 (N,C,32,32)。

        ★ 只对**本 carrier 用到的档位**做 8x 下采样 (见 _png_to_grid 的 sel),
          避免把 12 档全量展开成 float 中间张量。
        """
        keys = self.png_keys
        if keys is None:
            return None

        want_skel = self.carrier in ("dilate_skel", "both")
        want_blur = self.carrier in ("blur_gt", "both")
        sel = [i for i, k in enumerate(keys)
               if (want_skel and k.startswith("skel_w"))
               or (want_blur and k.startswith("blur_s"))]
        if not sel:
            return None
        avg = self._png_to_grid(png, sel=sel)
        pos = {g: j for j, g in enumerate(sel)}    # 全局 column -> avg 里的位置

        def _bucket_idx(glob_sel, values_px, want_px):
            """在 glob_sel 这些全局档里选与 want_px 最接近的 -> (N,) avg 内下标。"""
            v = torch.tensor(values_px, device=t.device)
            d = (want_px[:, None] - v[None, :]).abs()
            pick = d.argmin(dim=1)
            loc = [pos[g] for g in glob_sel]
            return torch.tensor(loc, device=t.device)[pick]

        def _gather(src, idx):
            return src.gather(
                1, idx.view(-1, 1, 1, 1).expand(-1, 1, *src.shape[-2:]))

        parts = []
        if want_skel:
            glob = [i for i, k in enumerate(keys) if k.startswith("skel_w")]
            if glob:
                k_t = self.sigma_sched.k_of(t)                 # (N,) **latent** px 半径
                # ★ 单位统一: k(t) latent px -> pixel 半径 (×8), 与预建 (w-1)/2 同域。
                #   不统一会静默饱和在最大档 (前几档永不可达)。
                k_px = k_t * self.png_downscale
                ws = [float(keys[i][len("skel_w"):]) for i in glob]  # pixel 笔宽
                idx = _bucket_idx(glob, [(w - 1.0) / 2.0 for w in ws], k_px)
                # 灰度 = 块内墨占比 ∝ 局部线宽, 保住粗细
                parts.append(_gather(avg, idx))
        if want_blur:
            glob = [i for i, k in enumerate(keys) if k.startswith("blur_s")]
            if glob:
                sig_t = self.sigma_sched.sigma_of(t)            # (N,) latent px σ
                sig_px = sig_t * self.png_downscale             # -> pixel σ 同域
                sigs = [float(keys[i][len("blur_s"):].replace("p", ".")) for i in glob]
                idx = _bucket_idx(glob, sigs, sig_px)
                parts.append(_gather(avg, idx))
        if not parts:
            return None
        out = parts[0] if len(parts) == 1 else 0.5 * (parts[0] + parts[1])
        # PNG 是单通道灰度; 复制到 latent_channels 以对齐 pred 的通道数
        return out.expand(-1, self.latent_channels, -1, -1)

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

    def __call__(self, pred_xstart, gt_lat, skel_lat, t, png=None):
        """pred_xstart: (N,C,32,32) 带梯度; gt_lat: GT 图像 latent; skel_lat: 细骨架 latent; t: (N,) flow t。

        png: 可选 (N,K,256,256) uint8 —— 预建载体 PNG。给了就走 png 通路 (省 GPU/免 encode),
             key 顺序须与 self.png_keys 一致 (即 dataset.mid_keys 的子序列)。
        ★ png 通路的下采样与 target 选择**不需要**梯度, 故整段 no_grad;
          pred_xstart 那侧仍带梯度正常回传。
        """
        zero = pred_xstart.sum() * 0.0
        if pred_xstart is None or t is None:
            return zero
        a_t = 1.0 - t.float()                       # 等效 sqrt_alpha (flow)
        active = (a_t >= self.alo) & (a_t <= self.ahi)
        if not bool(active.any()):
            return zero
        tgt = None
        if png is not None and self.png_keys:
            with torch.no_grad():
                tgt = self._target_png(png, t)
        if tgt is None:                             # 回退到 latent 域实时算子
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
