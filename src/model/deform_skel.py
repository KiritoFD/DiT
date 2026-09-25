#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeformSkel —— 用书家风格把**标准骨架 g** 形变成该书家的习惯间架。

## 为什么这个模块和前面所有风格模块不一样

前面那些（style_ln / spatial_film / 87-pair / style-rank）都失败了，根因是同一个：
**扩散 loss 不奖励风格**。它们必须靠 loss 之外的信号，而那个信号（strict ssim）量不到风格。

本模块不一样：
  实测 `g` 是**跨书家共享的规范字形**（43.5% 的字只有 1 张 std；字「出」11 个书家只用 3 张），
  而目标是**该书家写的那个字**。所以"把 g 形变到更接近目标"**直接降低重建 loss** ——
  它自带梯度。再加上出口的中间监督（逼近该书家的 GT 骨架），它有两个稠密信号。

## 网络：小 U-Net，输出**偏移场**而不是图像

  · 输出偏移场 -> 只做形变, **拓扑由 g 保证**, 不可能凭空画出别的字
  · 用 flow 模型是杀鸡用牛刀：目标是**确定性映射**，不是从噪声生成

## ★ 风格怎么用（v2 修正 —— v1 的核心缺陷）

v1 把风格**广播拼接**到 U-Net 输入，结果 `correct ≈ shuffled`：卷积层直接忽略了它，
只学到"从 g_std 到平均 g_gt"的通用形变。**"同一个 g_std 要产出不同书家的不同 g_gt"
这件事没被强制。**

v2 用三条路径强制风格进入：
  1. **FiLM 逐层调制**：风格 -> 每层 (γ,β)，直接调制 U-Net 特征（拼接可被忽略，FiLM 不能）
  2. **风格专属全局偏移场** `off_style(style)`：每个书家一张 (2,coarse,coarse) 的底图，
     表达"这个书家的整体间架倾向"（倾斜、长宽、重心）
  3. **内容自适应残差** `off_unet(g, style)`：在风格底图上做逐字修正
  -> `off = off_style + off_unet`，两者都受风格控制

## 其它关键设计点

  1. **低分辨率预测偏移 + 上采样**：32×32 逐像素偏移不连续，会把骨架撕碎
  2. **zero-init 输出层**：step0 恒等形变 -> 可安全 resume 做单变量 A/B
  3. **tanh 限幅**：±max_off 个 latent 像素（32 网格下 1 px ≈ 8 图像 px）
  4. **padding_mode='border'**：否则边界外补 0 -> 黑边伪影
  5. **base grid 必须用 align_corners=False 约定**（`(2*(i+0.5)/N)-1`）：
     用 `linspace(-1,1,N)` 会差半像素，零偏移时仍有半像素平移（实测 |out-g| 到 2.5）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class _FiLM(nn.Module):
    """风格 -> 逐层 (γ,β)。γ 以 1 为初值、β 以 0 为初值 -> step0 不改变特征。"""

    def __init__(self, cond_dim, n_ch):
        super().__init__()
        self.to_gb = nn.Linear(int(cond_dim), 2 * int(n_ch))
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        self.n_ch = int(n_ch)

    def forward(self, x, e):
        gb = self.to_gb(e)
        g, b = gb[:, :self.n_ch], gb[:, self.n_ch:]
        g = 1.0 + g
        return x * g[..., None, None] + b[..., None, None]


class DeformSkel(nn.Module):
    def __init__(self, cond_dim=128, ch=4, grid=32, style_ch=32,
                 width=64, max_off=3.0, coarse=8, residual=0, res_cap=1.0):
        super().__init__()
        self.grid = int(grid)
        self.coarse = int(coarse)
        self.max_off = float(max_off)
        self.style_proj = nn.Linear(int(cond_dim), int(style_ch))

        w = int(width)
        self.d1 = _block(ch + style_ch, w)          # grid
        self.d2 = _block(w, w * 2, stride=2)        # grid/2
        self.d3 = _block(w * 2, w * 2, stride=2)    # grid/4
        self.mid = _block(w * 2, w * 2)
        self.u2 = _block(w * 2 + w * 2, w)
        self.u1 = _block(w + w, w)
        # ★ 路径 1: FiLM 逐层调制（风格强制进入）
        self.f1 = _FiLM(cond_dim, w)
        self.f2 = _FiLM(cond_dim, w * 2)
        self.f3 = _FiLM(cond_dim, w * 2)
        self.fm = _FiLM(cond_dim, w * 2)
        # ★ 路径 3: 内容自适应残差偏移
        self.out = nn.Conv2d(w, 2, 1)
        # ★ 路径 4（可选）: 加性残差。2D 形变只能移动像素, **改不了笔画粗细/墨色**,
        #   而书家风格很大一部分正是这些 -> 只用形变时闭合率卡在 ~36%。
        #   残差头补上"形变做不到的那部分"。zero-init -> step0 仍恒等。
        self.residual = bool(residual)
        self.res = nn.Conv2d(w, ch, 1) if self.residual else None
        self.res_cap = float(res_cap)
        # ★ 路径 2: 风格专属**全分辨率**偏移底图（每个书家一张"习惯间架"）
        #   独立训练不受扩散稳定性约束 -> 风格可以激进注入, 不必压到 8x8 低分辨率。
        self.style_off = nn.Linear(int(cond_dim), 2 * self.grid * self.grid)
        # ★ 路径 5: 风格专属**全分辨率残差**（补上"形变改不了的"笔画粗细/墨色）
        self.style_res = nn.Linear(int(cond_dim), ch * self.grid * self.grid)             if bool(residual) else None
        for m in (self.out, self.style_off):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.res is not None:
            nn.init.zeros_(self.res.weight)
            nn.init.zeros_(self.res.bias)
        if self.style_res is not None:
            nn.init.zeros_(self.style_res.weight)
            nn.init.zeros_(self.style_res.bias)

        idx = (2.0 * (torch.arange(self.grid) + 0.5) / self.grid) - 1.0
        gy, gx = torch.meshgrid(idx, idx, indexing='ij')
        self.register_buffer('base_grid',
                             torch.stack([gx, gy], -1).unsqueeze(0), persistent=False)
        self.last_out = None
        self.last_mask = None   # 哪些样本真的做了形变
        self.last_off = None
        self.last_off_style = None
        self.last_res = None

    def forward(self, g, style):
        """g: (N,4,H,W) 标准骨架 latent；style: (N,cond_dim) 书家风格。返回 g' 同形状。"""
        n, c, h, w = g.shape
        if (h, w) != (self.grid, self.grid):
            g = F.interpolate(g, size=(self.grid, self.grid), mode='bilinear',
                              align_corners=False)
        st = style.float()
        s = self.style_proj(st)[..., None, None].expand(-1, -1, self.grid, self.grid)
        x0 = torch.cat([g.float(), s], 1)

        e1 = self.f1(self.d1(x0), st)
        e2 = self.f2(self.d2(e1), st)
        e3 = self.f3(self.d3(e2), st)
        m = self.fm(self.mid(e3), st)
        u = F.interpolate(m, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        u = self.u2(torch.cat([u, e2], 1))
        u = F.interpolate(u, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        u = self.u1(torch.cat([u, e1], 1))

        off_u = self.out(u)                                    # (N,2,grid,grid)
        if self.coarse > 0 and self.coarse < self.grid:
            off_u = F.interpolate(off_u, size=(self.coarse, self.coarse),
                                  mode='bilinear', align_corners=False)
            off_u = F.interpolate(off_u, size=(self.grid, self.grid),
                                  mode='bilinear', align_corners=False)
        off_s = self.style_off(st).view(n, 2, self.grid, self.grid)   # 全分辨率
        off = torch.tanh((off_u + off_s) / max(self.max_off, 1e-6)) * self.max_off

        grid = self.base_grid + off.permute(0, 2, 3, 1) / (self.grid / 2.0)
        g2 = F.grid_sample(g.float(), grid, mode='bilinear',
                           padding_mode='border', align_corners=False)
        if self.res is not None:
            r = self.res(u)
            if self.coarse > 0 and self.coarse < self.grid:
                r = F.interpolate(r, size=(self.grid, self.grid), mode='bilinear',
                                  align_corners=False)
            if self.style_res is not None:
                r = r + self.style_res(st).view(n, self.res.out_channels,
                                                self.grid, self.grid)
            r = torch.tanh(r / max(self.res_cap, 1e-6)) * self.res_cap
            g2 = g2 + r
            self.last_res = r.detach()
        self.last_out = g2
        self.last_off = off.detach()
        self.last_off_style = off_s.detach()
        return g2

    def offset_stats(self):
        """诊断用：偏移场幅值。mean|off| ≈ 0 说明形变没学动（退化成恒等）。"""
        if getattr(self, 'last_off', None) is None:
            return None
        o = self.last_off
        os_ = self.last_off_style
        d = dict(mean_abs=float(o.abs().mean()), max_abs=float(o.abs().max()),
                 cap=self.max_off,
                 style_part=float(os_.abs().mean()) if os_ is not None else 0.0)
        if getattr(self, 'last_res', None) is not None:
            d['res'] = float(self.last_res.abs().mean())
        return d
