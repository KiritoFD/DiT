# -*- coding: utf-8 -*-
"""`heun_batch` 的 **3-NFE 陷阱**回归守卫（`docs/infra_1015.md` §六 ✓）

审查原文:
  "你们已经算过一次 `v1`，第二个 `2B` forward 又重新算了一次 `v1_`。" ✓
  | 正常 Heun | B + B | 2 |
  | 当前 heun_batch | B + 2B | **3** | ✓
  "当前 `heun_batch=True` 的理论网络计算量比普通 Heun 多约 **50%**，
   而且**并没有减少 forward 调用次数**。" ✓
  "建议: 将当前实现标记为实验性/废弃；默认保留普通 Heun；
   **删除文档里"把两个 RK stage 合并成一次 forward"的错误描述**" ✓

★ 本测试守的**不变量**（最直接 ✓）:
  ① `heun_batch=True` 与 `False` 的 **forward 调用次数相同** ✓
  ② 且**每次的 batch 大小相同**（都是 B ✓，**没有** 2B ✗）
  ③ 且 **B-equivalent NFE = 2** ✓（不是 3 ✗）
  ⇒ 等价于断言"该开关已成为 no-op" ✓ —— 这正是修复的目标 ✓
"""
import warnings

import torch as th

from src.loss.flow_matching import FlowMatching


class _CountingModel(th.nn.Module):
    """极小模型 ✓ 记录每次 forward 的 batch 大小 ✓（只为数 NFE ✓）。"""

    def __init__(self, ch=4):
        super().__init__()
        self.w = th.nn.Parameter(th.zeros(1, ch, 1, 1))
        self.batches = []

    def forward(self, x, t, **kw):
        self.batches.append(int(x.shape[0]))
        return self.w.expand_as(x)


def _run(heun_batch, B=8, steps=4):
    fm = FlowMatching(num_steps=steps, sampler="heun", heun_batch=heun_batch)
    m = _CountingModel()
    x = th.randn(B, 4, 8, 8)
    with th.no_grad():
        if heun_batch:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fm.ddim_sample_loop(m, (B, 4, 8, 8), x_T=x, device="cpu")
        else:
            fm.ddim_sample_loop(m, (B, 4, 8, 8), x_T=x, device="cpu")
    return m.batches, B, steps


def test_heun_batch_has_no_2B_forward():
    """★ 不得出现 batch = 2B 的 forward ✗（旧实现的 3-NFE 根源 ✓）。"""
    batches, B, _steps = _run(True)
    assert batches, "没有捕获到任何 forward ✗ 测试无效 ✗"
    assert 2 * B not in batches, (
        "发现 batch=%d（=2B ✗）的 forward ✗ ⇒ 3-NFE 陷阱**回归** ✗\n"
        "（`docs/infra_1015.md` §六 ✓ 该分支必须复用已算出的 v1 ✓）实际 batches=%s"
        % (2 * B, batches))


def test_heun_batch_nfe_equals_normal_heun():
    """★ NFE 必须与普通 Heun **完全一致** ✓（= 该开关是 no-op ✓）。"""
    b_on, B, steps = _run(True)
    b_off, _s, _ = _run(False)
    assert b_on == b_off, (
        "heun_batch=True 的 forward 序列 %s ≠ False 的 %s ✗\n"
        "⇒ 开关语义不一致 ✗（infra_1015 §六 要求两者一致 ✓）" % (b_on, b_off))
    # ★ 每步 NFE = 总 forward 次数 / 步数 / B ✓（上一版漏了除步数 ✗）
    nfe = sum(b_on) / float(B) / float(steps)
    assert abs(nfe - 2.0) < 1e-9, (
        "B-equivalent NFE = %.3f ✗ 应为 **2** ✓（正常 Heun ✓）\n"
        "（§六: 旧实现是 3 ✗ = B + 2B ✗）实际 batches=%s steps=%d" % (nfe, b_on, steps))


def test_heun_batch_emits_deprecation_warning():
    """★ 必须**显式告警** ✓ —— 不静默改变数值 ✓（良好工程 ✓）。"""
    fm = FlowMatching(num_steps=2, sampler="heun", heun_batch=True)
    m = _CountingModel()
    x = th.randn(4, 4, 8, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with th.no_grad():
            fm.ddim_sample_loop(m, (4, 4, 8, 8), x_T=x, device="cpu")
    msgs = [str(x.message) for x in w if "heun_batch" in str(x.message)]
    assert msgs, "heun_batch=True 未发出废弃警告 ✗（§六 要求显式标注 ✓）"


def test_warning_is_emitted_once_only():
    """★ 警告只发一次 ✓（避免每步刷屏 ✗ —— 采样循环里会调用很多次 ✓）。"""
    fm = FlowMatching(num_steps=3, sampler="heun", heun_batch=True)
    m = _CountingModel()
    x = th.randn(4, 4, 8, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with th.no_grad():
            fm.ddim_sample_loop(m, (4, 4, 8, 8), x_T=x, device="cpu")
    n = len([1 for x_ in w if "heun_batch" in str(x_.message)])
    assert n == 1, "废弃警告应恰好发 1 次 ✓，实际 %d 次 ✗" % n
