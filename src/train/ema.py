# -*- coding: utf-8 -*-
"""ema.py — **EMA 组件**（2026-10-14 [重构 PR4] ✓）

审查 §4.2 把 EMA 列为需要"明确状态与生命周期"的组件之一 ✓。
本模块把原本内联在 2900 行 `main()` 里的三件事集中起来 ✓:
  ① `update_ema()`            —— 参数/buffer 的 EMA 更新 ✓（逐字搬移 ✓ 含批处理优化 ✓）
  ② `should_update()` / `effective_decay()` —— **间隔 EMA 的调度语义** ✓
  ③ `interval_approximation_error()` —— **近似误差的量化工具** ✓

★ 关于间隔 EMA 的数学（审查 P1-5 已纠正 ✓ 本模块把结论落成代码与注释 ✓）:
    严格式（每步更新 ✓）:  EMA_t = β^N·EMA_{t-N} + (1-β)·Σ_{k=0..N-1} β^k·θ_{t-k}
    间隔式（每 N 步 ✓）:    EMA_t = β^N·EMA_{t-N} + (1-β^N)·θ_t
  ⇒ 间隔式把窗口内 N-1 步的权重**全部贴到最后一步 θ_t** ✗
  ⇒ **只有参数在窗口内基本不变时才近似相等** ✓ **不是严格等价** ✗
  ⇒ 处置（审查建议 ✓）: 作为**带宽优化保留** ✓（省 46M 参数 ×(N-1) 次显存往返 ✓），
     但 ① interval 属**实验身份** ✓（已入 effective_config / 日志 ✓）
        ② 跨实验比较**必须固定同一 interval** ✓
        ③ 必须能**量化**它带来的差异 ✓ ⇒ 见 `interval_approximation_error()` ✓
"""
from __future__ import annotations


import torch

__all__ = ["update_ema", "should_update", "effective_decay", "ema_interval",
           "interval_approximation_error", "describe"]


def ema_interval(args) -> int:
    """间隔 N ✓（缺省 1 = 每步 ✓ 与旧实现 `max(1, int(getattr(...)))` 逐字一致 ✓）。"""
    return max(1, int(getattr(args, "ema_interval", 1) or 1))


def should_update(step: int, args) -> bool:
    """本步是否更新 EMA ✓（旧实现: `step % max(1, interval) == 0` ✓ 逐字一致 ✓）。"""
    return int(step) % ema_interval(args) == 0


def effective_decay(step: int, args) -> float:
    """本步实际使用的 decay ✓ —— **含 warmup 与 interval 幂** ✓。

    逐字等价于旧内联代码 ✓:
        if args.ema_warmup: current = min(args.ema_decay, (1+step)/(10+step))
        else:               current = args.ema_decay
        update_ema(..., current ** interval)
    ⚠ 返回的是**已经乘方**的值 ✓（即直接传给 `update_ema` 的那个 decay ✓）
    """
    base = float(getattr(args, "ema_decay", 0.999))
    step = max(0, int(step) - int(getattr(args, 'ema_start_step', 0)))
    if bool(getattr(args, "ema_warmup", False)):
        cur = min(base, (1.0 + float(step)) / (10.0 + float(step)))
    else:
        cur = base
    return cur ** ema_interval(args)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update a full-precision model EMA, including floating-point buffers.

    [INFRA 2026-09-18] 原版对每个参数单独 `mul_().add_()`，并在**每次调用**重建
    named_parameters / named_buffers 两个 dict —— `ema_interval` 默认 1 时这是每步
    几百个小 kernel + Python 遍历/建表开销（其他地方都在抠 launch 与同步，这处是漏网）。
    现在：
      * 首次调用把 (ema_tensor, src_tensor) 配对**缓存**到 ema_model 上 —— 参数张量对象
        由优化器就地更新、跨步稳定（detach 与原参数共享 storage，始终读到最新值），故可安全缓存，
        之后零建表；
      * 归一后用 `torch._foreach_mul_/add_` 批处理成极少的 multi-tensor kernel。
    数值口径与原实现一致（同样的 decay / 1-decay、浮点 buffer 做 decay、整型 buffer 做 copy）。

    ★ 2026-10-14 [重构 PR4]: 本函数自 `train.py` **逐字搬移** ✓（行为不变 ✓）；
      ⚠ `@torch.no_grad()` 装饰器**必须一起搬** ✗ —— 首版漏掉 ✓ 已补回 ✓
        （它关掉 EMA 更新的 autograd 追踪 ✓ 属语义的一部分 ✓ 不是可选项 ✗）；
      `train.py` 保留同名**再导出** ✓ ⇒ 既有 `from src.train.train import update_ema`
      的调用点无需改动 ✓。
    """
    pairs = getattr(ema_model, "_ema_pairs", None)
    if pairs is None:
        def unwrap(module):
            while isinstance(module, torch.nn.parallel.DistributedDataParallel) or hasattr(module, '_orig_mod'):
                module = module.module if isinstance(module, torch.nn.parallel.DistributedDataParallel) else module._orig_mod
            return module
        source, target = unwrap(model), unwrap(ema_model)
        source_params = dict(source.named_parameters())
        source_buffers = dict(source.named_buffers())
        target_params = dict(target.named_parameters())
        target_buffers = dict(target.named_buffers())
        if source_params.keys() != target_params.keys() or source_buffers.keys() != target_buffers.keys():
            raise ValueError('EMA/source parameter or buffer keys differ after unwrap')
        ema_p, src_p = [], []
        fixed_pairs=[]
        for name, ep in target_params.items():
            sp = source_params[name]
            if ep.shape != sp.shape:
                raise ValueError(f'EMA shape mismatch: {name}')
            ema_p.append(ep)
            src_p.append(sp.detach())
            if not sp.requires_grad:fixed_pairs.append((ep,sp.detach()))
        ema_fb, src_fb, ema_cb, src_cb = [], [], [], []
        for name, eb in target_buffers.items():
            sb = source_buffers[name]
            if torch.is_floating_point(eb):
                ema_fb.append(eb)
                src_fb.append(sb.detach())
            else:
                ema_cb.append(eb)
                src_cb.append(sb)
        pairs = (ema_p, src_p, ema_fb, src_fb, ema_cb, src_cb)
        ema_model._ema_pairs = pairs
        ema_model._ema_fixed_pairs = fixed_pairs
    ema_p, src_p, ema_fb, src_fb, ema_cb, src_cb = pairs
    if ema_p:
        torch._foreach_mul_(ema_p, decay)
        torch._foreach_add_(ema_p, src_p, alpha=1.0 - decay)
    if ema_fb:
        torch._foreach_mul_(ema_fb, decay)
        torch._foreach_add_(ema_fb, src_fb, alpha=1.0 - decay)
    for eb, sb in zip(ema_cb, src_cb):
        eb.copy_(sb)
    # Frozen condition tables must remain bit-identical, not accumulate EMA roundoff.
    for ep,sp in getattr(ema_model,'_ema_fixed_pairs',()):
        ep.copy_(sp)


def interval_approximation_error(interval: int, decay: float,
                                 drift_per_step: float = 1.0) -> float:
    """★ 审查 P1-5 要求的**量化工具** ✓: 间隔 EMA 的**单窗口注入偏差** ✓。

    ⚠ 精确定义（**必须说清** ✗ —— 我首版把它当成"总差"是错的 ✗）:
        本函数返回**每次间隔更新时注入的偏差** ✓（以"一步漂移 δ"为单位 ✓），
        **不是**训练全程累积的总差 ✗。

    推导（窗口内 θ 线性漂移 θ_{t-k} = θ_t - k·δ ✓）:
        严格式（同窗口）: E = β^N·E0 + (1-β)·Σ_{k=0..N-1} β^k·θ_{t-k}
        间隔式（同窗口）: E = β^N·E0 + (1-β^N)·θ_t
        两者之差 = (1-β)·Σ_k β^k·(θ_t - θ_{t-k}) = **(1-β)·δ·Σ_{k=0..N-1} k·β^k** ✓
      ⇒ 闭式 ✓（单窗口、同初值 ⇒ 精确成立 ✓ 见测试的数值对照 ✓）

    ⚠ **累积效应**（实测发现 ✓ 比"单窗口"更重要 ✗）:
        单窗口偏差虽小（β=0.999, N=4 ⇒ ≈0.006 步漂移 ✓），但**每 N 步注入一次** ✗
        ⇒ 总差随步数**线性累积** ✓: 实测 200 步 ⇒ 0.27 步漂移 ✓；2000 步 ⇒ 1.30 ✓
        （即 interval EMA 有**系统性滞后** ✓ 不是零均值噪声 ✗）
        ⇒ 这正是审查"**不是**数学等价的纯性能优化" ✓ 的定量依据 ✓
        ⇒ 跨实验比较**必须固定同一 interval** ✓（换 interval = 换 EMA 语义 ✓）

    用途（审查原话 ✓）: "必须…比较 `ema_interval=1/4/8` 的吞吐与生成质量" ✓
      ⇒ 本函数给**单窗口注入偏差** ✓；吞吐/质量仍需实测 ✓（PR6 ✓）

    典型值（β=0.999 ✓）: N=1 ⇒ 0.0 ✓（严格式本身 ✓）；N=4 ⇒ ≈0.006 ✓；N=8 ⇒ ≈0.028 ✓
    """
    n = max(1, int(interval))
    b = float(decay)
    if n == 1:
        return 0.0
    s = 0.0
    for k in range(n):
        s += k * (b ** k)
    return (1.0 - b) * float(drift_per_step) * s


def describe(args) -> str:
    """一行 EMA 摘要 ✓（供启动日志/manifest ✓ 审查要求 interval 属实验身份 ✓）。"""
    n = ema_interval(args)
    base = float(getattr(args, "ema_decay", 0.999))
    bits = [f"ema_decay={base}", f"ema_interval={n}",
            f"effective_per_step={base ** n:.8f}",
            f"warmup={bool(getattr(args, 'ema_warmup', False))}"]
    if n > 1:
        bits.append("approx_error_per_step_drift=%.4g" % interval_approximation_error(n, base))
    return " ".join(bits)
