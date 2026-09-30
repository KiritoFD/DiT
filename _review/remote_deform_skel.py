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


def _gauss_kernel(sigma, device, dtype):
    r = max(1, int(3.0 * float(sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    return (k[:, None] * k[None, :])[None, None]


def _up(x, size):
    """双线性重采样（放大/缩小都走这里）。

    ★★ 为什么不能直接用 F.interpolate(NCHW)：torch 2.5 的 NCHW
    `upsample_bilinear2d` CUDA 核在本机病态地慢 —— batch 2048 下
      C=192, 8x8 ->16x16 : 97.0 ms
      C=96, 16x16->32x32 : 49.5 ms
    两次加起来 147 ms/step，占了整个训练步的 30%（算子级 profiler 实测）。
    而同一个操作换成 channels_last 布局会走另一条向量化核：
      C=192: 97.0 -> 2.3 ms (42x)
      C=96 : 49.5 -> 4.8 ms (10x)
    数值上两者一致到 2.4e-07（纯 float32 舍入，已逐形状验证）。
    所以这里转布局 -> 插值 -> 转回来，**语义完全不变**，只是绕开慢核。
    """
    return F.interpolate(x.contiguous(memory_format=torch.channels_last),
                         size=size, mode='bilinear', align_corners=False) \
        .contiguous(memory_format=torch.contiguous_format)


def _block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class _FiLM(nn.Module):
    """风格 -> 逐层 (γ,β)。γ 以 1 为初值、β 以 0 为初值 -> step0 不改变特征。

    mode:
      "film"  : ``x·(1+γ) + β``            —— 旧行为（无归一化）
      "adaln" : ``LN(x)·(1+γ) + β``        —— [2026-09-29] 归一化后再调制。
        ⚠ 为什么加这个: 旧 FiLM **不归一化**, 调制效果取决于特征幅度; U-Net 浅层/深层
          特征幅度差异大, 同一个 Linear 产出的 (γ,β) 不可能在所有层都合适 ->
          **风格信号在部分层被淹没**。adaLN 让调制幅度与特征尺度解耦, 控制权完全
          交给风格向量（DiT 的标准做法）。且与 CalligStyleCrossAttn 内部的 pre-norm 一致。
        ⚠ 兼容: ``to_gb`` 的 Linear 形状不变 -> 已训 FiLM 权重**可原样载入**;
          新增的 norm 权重是新参数(会 missing), 但主体 warm-start 保留。
    """

    def __init__(self, cond_dim, n_ch, mode="film"):
        super().__init__()
        self.mode = str(mode)
        self.to_gb = nn.Linear(int(cond_dim), 2 * int(n_ch))
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)
        self.n_ch = int(n_ch)
        self.norm = nn.LayerNorm(int(n_ch)) if self.mode == "adaln" else None

    def forward(self, x, e):
        gb = self.to_gb(e)
        g, b = gb[:, :self.n_ch], gb[:, self.n_ch:]
        g = 1.0 + g
        if self.norm is not None:
            # ⚠ conv 特征是 (B,C,H,W), 末维是 W 不是 C —— nn.LayerNorm 默认按末维归一化
            #   会报 "expected input with shape [*, C]"。必须 permute 成 (B,H,W,C) 再归一化,
            #   即**逐空间位置、按通道维**做 LN。
            _x = x.permute(0, 2, 3, 1)
            _x = self.norm(_x)
            x = _x.permute(0, 3, 1, 2)
        return x * g[..., None, None] + b[..., None, None]


class DeformSkel(nn.Module):
    def __init__(self, cond_dim=128, ch=4, grid=32, style_ch=32,
                 width=64, max_off=3.0, coarse=8, residual=0, res_cap=1.0,
                 blur=0, blur_sigma=1.5, dt_ch=0, affine=1, ckpt=0,
                 stroke_mod=0, stroke_cap=1.0, gate_radius=0.25,
                 topo_mode=0, warp_iters=1,
                 film_mode="film", style_tokens=0, attn_heads=4,
                 preserve_amp=0):
        super().__init__()
        self.grid = int(grid)
        self.preserve_amp = bool(preserve_amp)
        # ★ 级联「小 warp」步数 (2026-09-29): 1=单次大 warp (旧行为, 默认);
        #   >1 时把总位移等分成 K 步依次重采样, 每步位移更小。见 forward 里的说明。
        self.warp_iters = max(1, int(warp_iters))
        self.coarse = int(coarse)
        self.max_off = float(max_off)
        self.style_proj = nn.Linear(int(cond_dim), int(style_ch))

        w = int(width)
        # ★ 输入加一路高斯模糊: 骨架 latent 很稀疏, U-Net 只看到细线时
        #   感受野里几乎没有梯度信号。模糊一份拼进去 -> 全图都有梯度。
        #   ⚠ 必须放在 d1 构造**之前**（res_cap 那一段在构造里更靠后, 放那儿会 UnboundLocalError）。
        self.blur = int(blur)
        self.blur_sigma = float(blur_sigma)
        # ★ 梯度检查点: U-Net 在 32x32/width96 下激活占大头(batch 4096 就吃满 23G),
        #   开了之后反向时重算前向 -> 激活内存降 2-3x, 代价是多算一遍(功耗反而上升)。
        self.ckpt = int(ckpt)
        self.dt_ch = int(dt_ch)
        self.use_affine = int(affine)
        _cin = ch + (ch if self.blur else 0) + self.dt_ch + style_ch
        self.d1 = _block(_cin, w)                   # grid
        self.d2 = _block(w, w * 2, stride=2)        # grid/2
        self.d3 = _block(w * 2, w * 2, stride=2)    # grid/4
        self.mid = _block(w * 2, w * 2)
        self.u2 = _block(w * 2 + w * 2, w)
        self.u1 = _block(w + w, w)
        # ★ 路径 1: FiLM 逐层调制（风格强制进入）
        #   film_mode="adaln" 时改为「归一化后再调制」(见 _FiLM 的说明)
        self.f1 = _FiLM(cond_dim, w, mode=film_mode)
        self.f2 = _FiLM(cond_dim, w * 2, mode=film_mode)
        self.f3 = _FiLM(cond_dim, w * 2, mode=film_mode)
        self.fm = _FiLM(cond_dim, w * 2, mode=film_mode)

        # ★ [2026-09-29] 路径 1b: **书家表经 cross-attention 注入**（用户要求的新增通路）
        #   动机: 单个 FiLM 向量只能做**全局**调制, 装不下"书家间架习惯"这种
        #   **局部、多模态**的风格。这里把风格向量展开成 K 个 token, 让**每个骨架位置**
        #   依自身内容+空间位置在 K 个 token 间做内容寻址聚合（与主模型 CalligStyleCrossAttn
        #   同一套机制, 直接复用）。
        #   位置: U-Net 瓶颈处 (grid/4 分辨率, w*2 通道) —— 最粗尺度、感受野最大,
        #   适合表达"整体间架"级别的风格。
        #   out_proj 是 zero-init -> **step0 恒等**, 不破坏 warm-start ✓
        self.style_tokens = int(style_tokens)
        self.attn = None
        self.style_tok_proj = None
        if self.style_tokens > 0:
            from .dit import CalligStyleCrossAttn
            _d = w * 2
            _g = max(2, self.grid // 4)
            self.style_tok_proj = nn.Linear(int(cond_dim), self.style_tokens * _d)
            self.attn = CalligStyleCrossAttn(_d, num_heads=int(attn_heads), grid_size=_g)
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
        # ★ 全局仿射: 书家风格 -> 2x3 矩阵, 先做一次全局缩放/拉伸/倾斜。
        #   "全局仿射定大局, 局部 offset 定细节" —— 欧阳询整体瘦长、颜真卿方正外拓
        #   这类整体倾向用 6 个参数就能表达, 不必让局部 U-Net 去硬凑。
        #   zero-init -> 恒等仿射 (theta=[1,0,0;0,1,0])。
        self.affine = nn.Linear(int(cond_dim), 6) if int(affine) else None
        if self.affine is not None:
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)
        self.style_off = nn.Linear(int(cond_dim), 2 * self.grid * self.grid)
        # ★ 路径 5: 风格专属**全分辨率残差**（补上"形变改不了的"笔画粗细/墨色）
        self.style_res = nn.Linear(int(cond_dim), ch * self.grid * self.grid)             if bool(residual) else None

        # ★ 路径 6: 受控笔画调制（Stroke Modulation）
        #   解决纯几何形变改不了粗细（墨量比 0.04）且避免自由残差在空白处凭空画线（作弊）。
        #   调制量被骨架笔画的局部邻域严格截断（gate），空白背景处严格为 0，绝对不破坏拓扑。
        self.use_stroke_mod = bool(stroke_mod)
        self.stroke_cap = float(stroke_cap)
        self.gate_radius = float(gate_radius)
        if self.use_stroke_mod:
            self.stroke_conv = nn.Conv2d(w, ch, 1)
            self.stroke_style = nn.Linear(int(cond_dim), ch * self.grid * self.grid)
            nn.init.zeros_(self.stroke_conv.weight)
            nn.init.zeros_(self.stroke_conv.bias)
            nn.init.zeros_(self.stroke_style.weight)
            nn.init.zeros_(self.stroke_style.bias)
        else:
            self.stroke_conv = None
            self.stroke_style = None

        for m in (self.out, self.style_off):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.res is not None:
            nn.init.zeros_(self.res.weight)
            nn.init.zeros_(self.res.bias)
        if self.style_res is not None:
            nn.init.zeros_(self.style_res.weight)
            nn.init.zeros_(self.style_res.bias)

        # 潜变量墨迹与背景物理基准向量:
        # SD VAE 下纯白背景 (255) 的潜变量均值: [2.18129, 1.42018, -0.00979, -1.14073]
        self.register_buffer("z_bg", torch.tensor([2.18129, 1.42018, -0.00979, -1.14073]).view(1, 4, 1, 1))
        # 墨迹 (black) 相对纯白背景的差分向量: [-3.16140, -4.00872, 1.12232, 2.44625]
        self.register_buffer("delta_ink", torch.tensor([-3.16140, -4.00872, 1.12232, 2.44625]).view(1, 4, 1, 1))

        # ★ 路径 7: 离散拓扑增删 (SkelNet-V2: 剪刀与胶水)
        #   剪刀 (head_prune): 预测省笔/减画空间掩码 -> 向白底背景 z_bg 插值
        #   胶水 (head_ligature): 预测牵丝/连带空间掩码 -> 沿墨迹 delta_ink 潜变量差分注入
        self.use_topo = bool(topo_mode)
        if self.use_topo:
            self.head_prune = nn.Conv2d(w, 1, 3, padding=1)
            self.head_ligature = nn.Conv2d(w, 1, 3, padding=1)
            # 初始化黄金法则: 卷积核全零, bias = -5.0 -> sigmoid(-5.0) ≈ 0.0067 ≈ 0
            # 保证 step 0 输出 100% 逐位等价于未加分支的旧模型
            nn.init.zeros_(self.head_prune.weight)
            nn.init.constant_(self.head_prune.bias, -5.0)
            nn.init.zeros_(self.head_ligature.weight)
            nn.init.constant_(self.head_ligature.bias, -5.0)
        else:
            self.head_prune = None
            self.head_ligature = None

        idx = (2.0 * (torch.arange(self.grid) + 0.5) / self.grid) - 1.0
        gy, gx = torch.meshgrid(idx, idx, indexing='ij')
        self.register_buffer('base_grid',
                             torch.stack([gx, gy], -1).unsqueeze(0), persistent=False)
        self.last_out = None
        self.last_mask = None   # 哪些样本真的做了形变
        self.last_off = None
        self.last_off_style = None
        self.last_res = None
        self.last_stroke_mod = None
        self.last_mask_prune = None
        self.last_mask_ligature = None

    def forward(self, g, style, dt=None):
        """g: (N,4,H,W) 标准骨架 latent；style: (N,cond_dim) 书家风格。返回 g' 同形状。"""
        n, c, h, w = g.shape
        if (h, w) != (self.grid, self.grid):
            g = _up(g, (self.grid, self.grid))
        st = style.float()
        s = self.style_proj(st)[..., None, None].expand(-1, -1, self.grid, self.grid)
        _parts = [g.float()]
        if self.dt_ch:
            if dt is None:
                # 在线自动极速距离场: 骨架在 GPU 上的多级膨胀距离近似
                _mag = g.abs().mean(dim=1, keepdim=True)
                _med = _mag.view(_mag.shape[0], -1).median(dim=1)[0].view(-1, 1, 1, 1)
                _mask = (_mag <= _med).float()
                _cur = _mask
                _dt = torch.zeros_like(_mask)
                for _i in range(1, 16):
                    _cur = F.max_pool2d(_cur, kernel_size=3, stride=1, padding=1)
                    _dt = torch.where((_dt == 0) & (_cur > 0) & (_mask == 0), float(_i), _dt)
                dt = _dt / 15.0
            _parts.append(dt.to(g.dtype))
        if self.blur:
            # 高斯模糊: 用 separable conv 实现, 无参数、可微、对 batch 广播
            _k = _gauss_kernel(self.blur_sigma, g.device, g.dtype)
            _b = F.conv2d(F.pad(g.float(), (_k.shape[-1] // 2,) * 4, mode='replicate'),
                          _k.expand(g.shape[1], 1, -1, -1), groups=g.shape[1])
            _parts.append(_b)
        _parts.append(s)
        x0 = torch.cat(_parts, 1)

        if self.ckpt and self.training:
            import torch.utils.checkpoint as _ck
            def _c(mod, film, x):
                return film(mod(x), st)
            e1 = _ck.checkpoint(_c, self.d1, self.f1, x0, use_reentrant=False)
            e2 = _ck.checkpoint(_c, self.d2, self.f2, e1, use_reentrant=False)
            e3 = _ck.checkpoint(_c, self.d3, self.f3, e2, use_reentrant=False)
            m = _ck.checkpoint(_c, self.mid, self.fm, e3, use_reentrant=False)
        else:
            e1 = self.f1(self.d1(x0), st)
            e2 = self.f2(self.d2(e1), st)
            e3 = self.f3(self.d3(e2), st)
            m = self.fm(self.mid(e3), st)
        # ★ [2026-09-29] 书家表 cross-attention 注入 (瓶颈处, 最粗尺度)
        #   Q = 瓶颈特征 token(+2D sincos 位置), K/V = K 个风格 token
        #   out_proj zero-init -> step0 恒等, 不破坏 warm-start
        if self.attn is not None:
            _B, _C, _H, _W = m.shape
            _q = m.flatten(2).transpose(1, 2)                      # (B, H*W, C)
            _sty = self.style_tok_proj(st).view(_B, self.style_tokens, _C)
            _q = self.attn(_q, _sty)                               # 残差 + zero-init
            m = _q.transpose(1, 2).reshape(_B, _C, _H, _W)
        u = _up(m, e2.shape[-2:])
        u = self.u2(torch.cat([u, e2], 1))
        u = _up(u, e1.shape[-2:])
        u = self.u1(torch.cat([u, e1], 1))

        off_u = self.out(u)                                    # (N,2,grid,grid)
        if self.coarse > 0 and self.coarse < self.grid:
            off_u = _up(off_u, (self.coarse, self.coarse))
            off_u = _up(off_u, (self.grid, self.grid))
        off_s = self.style_off(st).view(n, 2, self.grid, self.grid)   # 全分辨率
        off = torch.tanh((off_u + off_s) / max(self.max_off, 1e-6)) * self.max_off

        grid = self.base_grid + off.permute(0, 2, 3, 1) / (self.grid / 2.0)
        _d_aff = None
        if self.affine is not None:
            # ★ 先全局仿射, 再局部形变: 两者都作用在采样网格上(纯坐标变换, 保证拓扑)
            th = torch.tanh(self.affine(st)).view(-1, 2, 3)
            th = th + torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                                   device=th.device, dtype=th.dtype)
            # affine_grid 生成的是 (x_src, y_src) 采样坐标 -> 与 base_grid 同语义
            g_a = F.affine_grid(th, (n, 1, self.grid, self.grid), align_corners=False)
            _d_aff = g_a - self.base_grid
            grid = grid + _d_aff
        if self.warp_iters <= 1:
            g2 = F.grid_sample(g.float(), grid, mode='bilinear',
                               padding_mode='border', align_corners=False)
        else:
            # ★ 级联「小 warp」(2026-09-29): 把总位移等分成 K 步依次重采样。
            #   动机: 单次大位移 (max_off=6 ≈ 6 个 latent 像素) 的双线性重采样会把
            #   1px 宽的骨架线抹开 -> 解码发灰(实测 pred ink 只有 GT 的 1/4)。
            #   拆成 K 步后**每步位移 = 总位移/K**, 单步的亚像素插值误差更小。
            #   ⚠ 实验性开关: K 次重采样 = K 次双线性插值, 也会累加模糊, 是否更好必须实测。
            #   **K=1 时与旧行为逐位一致**(默认), 不影响任何已有 ckpt。
            _g = g.float()
            _step_off = off.permute(0, 2, 3, 1) / (self.grid / 2.0) / float(self.warp_iters)
            _step_aff = (None if _d_aff is None else _d_aff / float(self.warp_iters))
            for _k in range(self.warp_iters):
                _gk = self.base_grid + _step_off
                if _step_aff is not None:
                    _gk = _gk + _step_aff
                _g = F.grid_sample(_g, _gk, mode='bilinear',
                                   padding_mode='border', align_corners=False)
            g2 = _g

        # ★ 墨迹保幅校准 (Amplitude Calibration): 根除双线性采样导致的细线淡化与断裂
        if self.preserve_amp:
            _v = g.float() - self.z_bg
            _energy = _v.norm(dim=1, keepdim=True)
            _energy_peak = F.max_pool2d(_energy, kernel_size=3, stride=1, padding=1)
            _energy_peak_w = F.grid_sample(_energy_peak, grid, mode='bilinear', padding_mode='border', align_corners=False)
            _v_warped = g2 - self.z_bg
            _v_energy = _v_warped.norm(dim=1, keepdim=True)
            _scale = torch.clamp(_energy_peak_w / torch.clamp(_v_energy, min=1e-3), min=1.0, max=2.5)
            _gate = torch.clamp((_v_energy - 0.5) / 1.0, min=0.0, max=1.0)
            g2 = self.z_bg + _v_warped * (1.0 + _gate * (_scale - 1.0))

        if self.res is not None:
            r = self.res(u)
            if self.coarse > 0 and self.coarse < self.grid:
                r = _up(r, (self.grid, self.grid))
            if self.style_res is not None:
                r = r + self.style_res(st).view(n, self.res.out_channels,
                                                self.grid, self.grid)
            r = torch.tanh(r / max(self.res_cap, 1e-6)) * self.res_cap
            g2 = g2 + r
            self.last_res = r.detach()

        # ★ 受控笔画调制: 只在笔画局部邻域内增强/调粗细, 空白背景处 gate 严格为 0 (防作弊/防虚假笔画)
        if self.use_stroke_mod and self.stroke_conv is not None:
            if dt is not None:
                # 距离场采样到形变后坐标
                dt_w = F.grid_sample(dt.float(), grid, mode='bilinear',
                                     padding_mode='border', align_corners=False)
                gate = torch.clamp(1.0 - dt_w / max(self.gate_radius, 1e-4), min=0.0, max=1.0)
            else:
                # 备用: 无外部 dt 时按局部幅值软门控
                _mag = g2.abs().mean(dim=1, keepdim=True)
                gate = torch.clamp(_mag / 0.5, min=0.0, max=1.0)

            mod_u = self.stroke_conv(u)
            if self.coarse > 0 and self.coarse < self.grid:
                mod_u = _up(mod_u, (self.grid, self.grid))
            mod_s = self.stroke_style(st).view(n, c, self.grid, self.grid)
            mod = torch.tanh((mod_u + mod_s) / max(self.stroke_cap, 1e-6)) * self.stroke_cap
            g2 = g2 + gate * mod
            self.last_stroke_mod = (gate * mod).detach()
        else:
            self.last_stroke_mod = None

        # ★ 路径 7: 离散拓扑增删 (剪刀与胶水)
        if self.use_topo and self.head_prune is not None:
            m_prune = torch.sigmoid(self.head_prune(u))
            m_lig = torch.sigmoid(self.head_ligature(u))
            
            # 剪刀向白底潜变量 z_bg 插值 (干净抹除指定笔画, 杜绝灰黄泥斑)
            g2 = (1.0 - m_prune) * g2 + m_prune * self.z_bg
            # 胶水向墨迹潜变量方向拉升 (生成真实黑色游丝连线)
            g2 = g2 + m_lig * self.delta_ink
            
            self.last_mask_prune = m_prune
            self.last_mask_ligature = m_lig
        else:
            self.last_mask_prune = None
            self.last_mask_ligature = None

        self.last_out = g2
        self.last_off = off.detach()
        self.last_off_style = off_s.detach()
        return g2

    def regularizers(self):
        """TV 平滑 + Jacobian 折叠惩罚。都作用在**偏移场**上（纯几何量）。

        TV:       相邻像素偏移差 -> 防止把一根横线扭成波浪线
        Jacobian: det(I + ∇U) > 0 -> 防止空间折叠（笔画交叉处翻转）
                  返回 relu(-det) 的均值, 0 = 无折叠
        """
        if getattr(self, 'last_off', None) is None:
            return None
        u = self.last_off.float()                       # (N,2,32,32)
        du_dx = u[..., :, 1:] - u[..., :, :-1]
        du_dy = u[..., 1:, :] - u[..., :-1, :]
        tv = du_dx.pow(2).mean() + du_dy.pow(2).mean()
        # Jacobian: J = I + ∇U, 2x2。U 是 2 通道 (Ux, Uy)。
        #   a = ∂Ux/∂x, b = ∂Ux/∂y, c = ∂Uy/∂x, d = ∂Uy/∂y
        #   det = (1+a)(1+d) - b*c
        #   ⚠ 上一版把交叉项 b/c 直接写成 0 —— 那等于假设场是无旋的, 会漏掉真实折叠。
        #     这里用有限差分补齐。
        ux, uy = u[:, 0], u[:, 1]                       # (N,32,32)
        ax = ux[:, :, 1:] - ux[:, :, :-1]               # ∂Ux/∂x
        ay = ux[:, 1:, :] - ux[:, :-1, :]               # ∂Ux/∂y
        cx = uy[:, :, 1:] - uy[:, :, :-1]               # ∂Uy/∂x
        cy = uy[:, 1:, :] - uy[:, :-1, :]               # ∂Uy/∂y
        a = ax[:, :-1, :]; d = cy[:, :, :-1]
        b = ay[:, :, :-1]; c = cx[:, :-1, :]
        det = (1.0 + a) * (1.0 + d) - b * c
        fold = torch.relu(-det).mean()
        d = dict(tv=tv, fold=fold, det_min=float(det.min()))
        # ★ latent 空间正则（不需要 VAE）:
        #   实测破碎的来源不是幅度(g' 的 |mean| 只比目标低 6%), 而是空间分布。
        #   tv_out: 输出 latent 的梯度能量 —— 直接压"不该有的高频"
        #   tv_res: 加性残差的梯度能量 —— 残差是"凭空写像素"的唯一入口
        if self.last_out is not None:
            o = self.last_out.float()
            d['tv_out'] = ((o[..., :, 1:] - o[..., :, :-1]).pow(2).mean()
                           + (o[..., 1:, :] - o[..., :-1, :]).pow(2).mean())
        if getattr(self, 'last_res', None) is not None:
            r = self.last_res.float()
            d['tv_res'] = ((r[..., :, 1:] - r[..., :, :-1]).pow(2).mean()
                           + (r[..., 1:, :] - r[..., :-1, :]).pow(2).mean())
        if getattr(self, 'last_stroke_mod', None) is not None:
            sm = self.last_stroke_mod.float()
            d['tv_stroke'] = ((sm[..., :, 1:] - sm[..., :, :-1]).pow(2).mean()
                              + (sm[..., 1:, :] - sm[..., :-1, :]).pow(2).mean())
        if getattr(self, 'last_mask_prune', None) is not None:
            d['l1_prune'] = self.last_mask_prune.mean()
        if getattr(self, 'last_mask_ligature', None) is not None:
            d['l1_lig'] = self.last_mask_ligature.mean()
        return d

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
        if getattr(self, 'last_stroke_mod', None) is not None:
            d['stroke_mod'] = float(self.last_stroke_mod.abs().mean())
        return d
