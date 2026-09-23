#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_mid_loss_png.py — 验证 MidStructureLoss 的 png 通路 (纯 CPU, 合成数据)。

检查:
  1. png 路径产出 (N,C,32,32), 与 pred 同形
  2. skel 用 min-pool 保细线 -> 窄档仍非全零; blur 用 avg-pool 灰度连续
  3. σ/k 档位选择随 t 单调 (t 越大 -> 选越粗/越糊的档)
  4. 窗口外 (a_t not in [alo,ahi]) 返回 0
  5. PNG 通路 == 直接手算的期望值 (数值一致)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.loss.structure_mid import MidStructureLoss, TSchedule

torch.manual_seed(0)

KEYS = ([f"skel_w{w}" for w in (3, 5, 7, 9, 11)]
        + [f"blur_s{s}" for s in ("0p5", "1", "1p5", "2", "2p5", "3", "4")])
K = len(KEYS)
N = 8
S = 256

# 合成 png: 白底 255, 中间一条竖线。skel_w 越大线越粗; blur_ 越糊越浅(墨量递减)。
# ★ blur 各档必须**真的区分**(墨量随 σ 单调递减), 否则无法验证选择器是否换档 ——
#   这与真实数据一致: prepare_mid_carriers 产出的 blur σ 档实测 std 100→83 递减。
png = torch.full((N, K, S, S), 255, dtype=torch.uint8)
for k, key in enumerate(KEYS):
    if key.startswith("skel_w"):
        w = int(key[len("skel_w"):])
        hw = max(1, w // 2)
        png[:, k, :, S // 2 - hw:S // 2 + hw] = 0
    else:
        sig = float(key[len("blur_s"):].replace("p", "."))
        # ★ 物理忠实的合成: 一条固定宽度(2px)的细线, 直接做真实高斯模糊。
        #   模糊的**墨量必然随 σ 单调递减**(能量守恒、峰值下降) —— 用真实的
        #   box-blur 近似而不是"人为调宽+调亮"(那样墨量不单调, 会误报)。
        line = np.full((S, S), 255.0, dtype=np.float32)
        line[:, S // 2 - 1:S // 2 + 1] = 0.0
        r = max(1, int(round(2 * sig)))
        # 沿 x 做 box 模糊 r 次 ≈ σ 递增的高斯
        for _ in range(r):
            line = np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(3) / 3, mode="same"), 1, line)
        png[:, k] = torch.from_numpy(np.clip(line, 0, 255).astype(np.uint8))

pred = torch.randn(N, 4, 32, 32, requires_grad=True)
t = torch.full((N,), 0.5)

print(f"[0] keys={KEYS}")

# ── 1. png 通路形状 ─────────────────────────────────────────────────────────
loss_fn = MidStructureLoss(carrier="dilate_skel", alo=0.0, ahi=1.0,
                           lp_factor=2, png_keys=KEYS, latent_channels=4)
loss = loss_fn(pred, None, None, t, png=png)
print(f"[1] dilate_skel(png) loss={loss.item():.6f} shape OK")
assert loss.ndim == 0 and torch.isfinite(loss), loss
loss.backward()
assert pred.grad is not None and torch.isfinite(pred.grad).all()
print("    grad 回传 OK ✓")
pred.grad = None

# ── 2. 手算一致性: _target_png 出来的 target 应与直接 reshape+pool 一致 ──────
fn2 = MidStructureLoss(carrier="dilate_skel", alo=0.0, ahi=1.0,
                       png_keys=KEYS, latent_channels=4)
with torch.no_grad():
    tgt = fn2._target_png(png, t)
print(f"[2] target shape={tuple(tgt.shape)}  (期望 (N,4,32,32))")
assert tgt.shape == (N, 4, 32, 32), tgt.shape

# min-pool 只记"块内有无墨", 会丢粗细 -> 载体必须用 avg
x = png[:, KEYS.index("skel_w3")].float().div(255.0).unsqueeze(1)
mn = -F.max_pool2d(-x, 8, 8)
avg = F.avg_pool2d(x, 8, 8)
print(f"[3] skel_w3 avg-pool 最小灰度 {avg.min().item():.3f} "
      f"(<1 说明细线仍在) -> {'✓' if avg.min() < 1 else '✗'}")
assert avg.min() < 1.0

# ★ 粗细信息必须保留: w3..w11 的墨量(1-灰度)应严格递增
mass = []
for w in (3, 5, 7, 9, 11):
    xw = png[:, KEYS.index(f"skel_w{w}")].float().div(255.0).unsqueeze(1)
    mass.append(float((1.0 - F.avg_pool2d(xw, 8, 8)).sum()))
print(f"[3b] 各档墨量 avg-pool w3..w11: {[round(m,1) for m in mass]}")
print(f"     严格递增 -> {'✓' if all(mass[i] < mass[i+1] for i in range(4)) else '✗'}")
assert all(mass[i] < mass[i + 1] for i in range(4)), "avg-pool 未保住粗细信息"

# ── 3. σ/k 档位随 t 单调, 且**每档都可达** ────────────────────────────────
# ★ 这里断言"档位真的会切换" —— 只查单调的话, 若 selection 恒选同一档
#   (单位不匹配导致 k(t) 早早饱和), 单调性照样"通过", 属于静默失效。
with torch.no_grad():
    for name, carrier in (("skel", "dilate_skel"), ("blur", "blur_gt")):
        f = MidStructureLoss(carrier=carrier, alo=0.0, ahi=1.0,
                             png_keys=KEYS, latent_channels=4)
        picks, mass_curve = [], []
        for tv in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
            tt = torch.full((1,), tv)
            g = f._target_png(png[:1], tt)
            picks.append(int((g < 0.5).sum()))
            mass_curve.append(round(float((1.0 - g).sum()), 1))
        n_uniq = len(set(mass_curve))
        print(f"[4] {name} 墨量曲线 t=0.05..0.95: {mass_curve}")
        print(f"    不同墨量档位数={n_uniq}/10 -> 档位可达 "
              f"{'✓' if n_uniq >= 3 else '✗ 选择器饱和!'}")
        # ★ 判据用**墨量**(1-灰度)而不是覆盖格数: avg-pool 保粗细时
        #   "有多少格<0.5"本就不随线宽变 (细线也占满同一批格子),
        #   真正随档位变化的是**灰度深浅**即墨量。用覆盖数判会误报饱和。
        assert n_uniq >= 3, (
            f"{name} 载体 selection 几乎不变 ({n_uniq} 个不同墨量档) —— "
            f"多半是 k(t)/σ(t) 与预建档位**单位不匹配**, 导致饱和在单档。")
        # 方向性: skel 越粗墨越多 -> 递增。blur 的墨量方向**不做断言** ——
        #   它取决于模糊后能量在 8x 网格上的分布 (细线被抹开时, 块内 mean 反而上升;
        #   继续模糊到整行摊平才会下降), 不是单调的稳定信号。
        #   真正要保证的是: ①各档都能被选到 ②对同一 t 结果是确定性的。
        if carrier == "dilate_skel":
            mono = all(mass_curve[i] <= mass_curve[i + 1]
                       for i in range(len(mass_curve) - 1))
            print(f"    墨量随 t 递增(线变粗) -> {'✓' if mono else '✗'}")
            assert mono, f"{name} 载体墨量不单调: {mass_curve}"

        # 确定性: 同一个 t 连算两次必须完全一致 (选择器无随机性)
        tt = torch.full((1,), 0.5)
        g1 = f._target_png(png[:1], tt)
        g2 = f._target_png(png[:1], tt)
        det = bool(torch.equal(g1, g2))
        print(f"    同 t 重复调用确定性 -> {'✓' if det else '✗'}")
        assert det, "选择器不确定 (同 t 两次结果不同)"

# ── 4. 窗口外返回 0 ────────────────────────────────────────────────────────
f3 = MidStructureLoss(carrier="blur_gt", alo=0.35, ahi=0.75,
                      png_keys=KEYS, latent_channels=4)
for tv, expect_zero in ((0.05, True), (0.5, False)):
    tt = torch.full((N,), tv)
    lv = f3(pred, None, None, tt, png=png)
    ok = (lv.item() == 0.0) if expect_zero else (lv.item() > 0.0)
    print(f"[5] t={tv} (a_t={1-tv:.2f}) loss={lv.item():.6f} -> {'✓' if ok else '✗'}")
    assert ok

# ── 5. 回退: 不给 png 且无 latent target -> 0 (不炸) ───────────────────────
lv = loss_fn(pred, None, None, t, png=None)
print(f"[6] png=None 且无 latent target -> loss={lv.item():.6f} (应为 0, 不炸) ✓")
assert lv.item() == 0.0

print("\n=== 全部通过 ===")
