# -*- coding: utf-8 -*-
"""test_refactor_pr4c.py —— 2026-10-14 [重构 PR4] **EMA 组件** 的验收测试 ✓

审查 §4.2 把 EMA 列为需要明确生命周期的组件之一 ✓
审查 P1-5: "必须纠正注释、把间隔写入实验身份，并**比较 `ema_interval=1/4/8` 的吞吐与
  生成质量**。不能将其称为数学等价的纯性能优化" ✓

本组测试锁定:
  ① `update_ema` 与**朴素参考实现**数值逐位一致 ✓（含浮点/整型 buffer 两种口径 ✓）
  ② `should_update` / `effective_decay` 与**旧内联表达式**逐字等价 ✓（含 warmup ✓）
  ③ ★ 近似误差的**闭式解**与**数值模拟**互相印证 ✓（P1-5 的量化依据 ✓）
  ④ ★ `@torch.no_grad()` **必须随函数一起搬** ✗（我首版漏掉 ⇒ 已被本测试锁死 ✓）
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import ema as EM                                            # noqa: E402
from src.train import train as TR                                          # noqa: E402


# ─────────────── ① update_ema 数值等价 ───────────────
class _M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)
        self.register_buffer("fb", torch.randn(5))          # 浮点 buffer ✓
        self.register_buffer("ib", torch.arange(6))         # 整型 buffer ✓


def _reference_update(ema_model, model, decay):
    """**朴素参考实现** ✓（逐参数 mul_/add_ ✓ 与优化前的原始实现同口径 ✓）。"""
    src = model.module if hasattr(model, "module") else model
    sp = dict(src.named_parameters())
    sb = dict(src.named_buffers())
    with torch.no_grad():
        for name, ep in ema_model.named_parameters():
            if name in sp:
                ep.mul_(decay).add_(sp[name].detach(), alpha=1.0 - decay)
        for name, eb in ema_model.named_buffers():
            if name not in sb:
                continue
            if torch.is_floating_point(eb):
                eb.mul_(decay).add_(sb[name].detach(), alpha=1.0 - decay)
            else:
                eb.copy_(sb[name])


@pytest.mark.parametrize("decay", [0.9, 0.999, 0.9999])
def test_update_ema_matches_reference(decay):
    torch.manual_seed(0)
    src = _M()
    a = _M()
    a.load_state_dict(src.state_dict())
    b = _M()
    b.load_state_dict(src.state_dict())
    # 让源参数离开初值 ✓
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        src.fb.add_(1.0)
        src.ib.add_(7)
    EM.update_ema(a, src, decay)
    _reference_update(b, src, decay)
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"参数 {n1} 不等 ✗"
    for (n1, p1), (n2, p2) in zip(a.named_buffers(), b.named_buffers()):
        assert n1 == n2 and torch.equal(p1, p2), f"buffer {n1} 不等 ✗"


def test_update_ema_int_buffer_is_copied():
    torch.manual_seed(1)
    src = _M()
    a = _M()
    a.load_state_dict(src.state_dict())
    with torch.no_grad():
        src.ib.add_(100)
    EM.update_ema(a, src, 0.5)
    assert torch.equal(a.ib, src.ib), "整型 buffer 必须**直接 copy** ✓（不衰减 ✗）"
    assert a.ib.dtype == src.ib.dtype


def test_update_ema_caches_pairs_once():
    """配对缓存 ✓（首次建表, 之后复用 ✓ —— 旧 INFRA 优化的语义 ✓）"""
    torch.manual_seed(2)
    src = _M()
    a = _M()
    a.load_state_dict(src.state_dict())
    EM.update_ema(a, src, 0.9)
    pairs1 = a._ema_pairs
    EM.update_ema(a, src, 0.9)
    assert a._ema_pairs is pairs1, "第二次调用必须复用缓存 ✓（不得重建 ✗）"


def test_train_reexports_same_object():
    """★ 兼容性 ✓: `train.update_ema` 必须是**同一函数对象** ✓（不是副本 ✗）"""
    assert TR.update_ema is EM.update_ema, \
        "train.update_ema 与 ema.update_ema 必须同一对象 ✗（再导出而非复制 ✓）"


def test_update_ema_keeps_no_grad_decorator():
    """★★ 锁死我踩过的坑 ✗: `@torch.no_grad()` 必须随函数一起搬 ✓

    首版搬移时漏掉了它 ✗（同时 train.py 里留下**孤儿装饰器** ⇒ SyntaxError ✗）
    ⇒ 本测试把它钉住 ✓
    """
    src = inspect.getsource(EM.update_ema)
    assert "@torch.no_grad()" in src, \
        "update_ema 丢失 @torch.no_grad() ✗（它关掉 EMA 更新的 autograd 追踪 ✓ 属语义 ✓）"
    # 且 train.py 不得有孤儿装饰器 ✗
    import io
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "@torch.no_grad()\nfrom src.train.ema import" not in t, \
        "train.py 残留孤儿装饰器 ✗（会导致 SyntaxError ✗）"


# ─────────────── ② 调度语义逐字等价 ───────────────
class _A:
    def __init__(self, **kw):
        self.ema_decay = kw.pop("ema_decay", 0.999)
        self.ema_interval = kw.pop("ema_interval", 1)
        self.ema_warmup = kw.pop("ema_warmup", False)


def _old_should(step, a):
    return step % max(1, int(getattr(a, "ema_interval", 1))) == 0


def _old_decay(step, a):
    if a.ema_warmup:
        cur = min(a.ema_decay, (1.0 + step) / (10.0 + step))
    else:
        cur = a.ema_decay
    return cur ** max(1, int(getattr(a, "ema_interval", 1)))


@pytest.mark.parametrize("interval", [1, 2, 4, 8])
@pytest.mark.parametrize("warmup", [False, True])
def test_schedule_matches_old_inline(interval, warmup):
    a = _A(ema_interval=interval, ema_warmup=warmup)
    for step in (0, 1, 3, 4, 7, 8, 100, 1000):
        assert EM.should_update(step, a) == _old_should(step, a), step
        assert EM.effective_decay(step, a) == _old_decay(step, a), step


def test_schedule_handles_missing_and_bad_interval():
    class _Bare:
        pass
    a = _Bare()
    assert EM.ema_interval(a) == 1 and EM.should_update(0, a)
    a2 = _A(ema_interval=0)                     # 0 ⇒ 夹到 1 ✓（旧实现 max(1,..) ✓）
    assert EM.ema_interval(a2) == 1
    a3 = _A(ema_interval=None)
    assert EM.ema_interval(a3) == 1


# ─────────────── ③ ★ 闭式解 vs 数值模拟 ───────────────
def test_interval_error_closed_form_matches_single_window():
    """★★ P1-5 的量化依据 ✓: 闭式 = **单窗口**注入偏差 ✓（精确对照 ✓）

    对照方式（**同初值、同一窗口** ✓ —— 这样才与闭式的定义一致 ✓）:
      严格式: E = β^N·E0 + (1-β)·Σ_{k=0..N-1} β^k·θ_{t-k} ✓
      间隔式: E = β^N·E0 + (1-β^N)·θ_t ✓
      差   = (1-β)·δ·Σ k·β^k ✓ = 闭式 ✓

    ⚠ 我首版用"多步累积"对照 ⇒ 量级差 45 倍 ✗ —— **不是闭式错** ✗
      而是两者**定义不同** ✓: 闭式是**单窗口注入** ✓；多步模拟含**累积** ✓
      （实测: 200 步 ⇒ 0.27 步漂移 ✓；2000 步 ⇒ 1.30 ✓ ⇒ 线性累积 ✓）
      ⇒ 已把定义写进 ema.py 文档 ✓ 并改为单窗口对照 ✓
    """
    beta, delta, N = 0.999, 1.0, 4
    theta_t = 10.0
    # 同一窗口内的 θ 序列: θ_{t-k} = θ_t - k·δ ✓
    win = [theta_t - k * delta for k in range(N)]        # [θ_t, θ_{t-1}, ..., θ_{t-N+1}] ✓
    E0 = 3.0
    strict = (beta ** N) * E0 + (1 - beta) * sum(
        (beta ** k) * win[k] for k in range(N))
    inter = (beta ** N) * E0 + (1 - beta ** N) * theta_t
    sim = abs(strict - inter)
    closed = EM.interval_approximation_error(N, beta, delta)
    assert closed > 0 and sim > 0
    assert abs(sim - closed) < 1e-12, \
        f"单窗口闭式必须**精确**吻合 ✗ closed={closed!r} sim={sim!r}"


def test_interval_error_accumulates_linearly():
    """★ 实测发现的**累积效应** ✓（比单窗口更重要 ✗ —— 写进测试防误读 ✓）

    间隔 EMA 每 N 步注入一次**同号**偏差 ⇒ 总差随步数**线性增长** ✓
    ⇒ 它不是零均值噪声 ✗ 而是**系统性滞后** ✓
    """
    beta, delta, N = 0.999, 1.0, 4

    def _total(steps):
        strict = inter = 0.0
        for t in range(steps):
            strict = beta * strict + (1 - beta) * (t * delta)
            if (t + 1) % N == 0:
                inter = (beta ** N) * inter + (1 - beta ** N) * (t * delta)
        return abs(strict - inter)

    e200, e2000 = _total(200), _total(2000)
    assert e2000 > e200, (e200, e2000)
    assert e2000 / e200 > 3.0, "总差应随步数显著增长 ✓（线性累积 ✓）"
    # 且**远大于**单窗口注入偏差 ✓（说明二者定义不同 ✓ 不得混用 ✓）
    assert e200 > 10 * EM.interval_approximation_error(N, beta, delta)


def test_interval_error_zero_for_n1():
    assert EM.interval_approximation_error(1, 0.999) == 0.0, \
        "N=1 就是严格式 ⇒ 误差必须为 0 ✓"


def test_interval_error_grows_with_interval():
    vals = [EM.interval_approximation_error(n, 0.999) for n in (1, 2, 4, 8, 16)]
    assert vals == sorted(vals) and vals[0] == 0.0, vals
    assert vals[3] > vals[2] > 0


def test_describe_reports_identity():
    a = _A(ema_interval=4)
    d = EM.describe(a)
    for token in ("ema_decay=0.999", "ema_interval=4", "effective_per_step=",
                  "warmup=False", "approx_error_per_step_drift="):
        assert token in d, (token, d)
    d1 = EM.describe(_A(ema_interval=1))
    assert "approx_error" not in d1, "N=1 不该报近似误差 ✓（它就是严格式 ✓）"


def test_train_py_uses_ema_component():
    """接线守卫 ✓: 循环里必须走组件 ✓ 且启动日志打印 EMA 身份 ✓"""
    import io
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_ema.should_update(" in t and "_ema.effective_decay(" in t, "未接线 ✗"
    assert "[ema]" in t, "启动日志必须打印 EMA 摘要 ✓（interval 属实验身份 ✓）"
    assert "current_ema_decay ** max(1, int(getattr(args, 'ema_interval'" not in t, \
        "旧内联幂运算仍在 ✗（应走 effective_decay ✓）"
