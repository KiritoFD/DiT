# -*- coding: utf-8 -*-
"""telemetry.py — **训练期诊断（纯观测）**（2026-10-14 [重构 PR4] ✓）

审查 §3 目标结构里有 `training/telemetry.py` ✓；审查 §6.3 要求性能监控分三层 ✓
（吞吐 / GPU 时间 / 硬件利用 ✓）—— 本模块落地其中**已有的两件观测工具** ✓:

  ① `StepProfiler`           —— 算子级 profile（窗口 + **逐算子** CUDA/CPU 时间表 ✓）
      用途（原注释）: "回答 GPU 时间花在哪些算子上、还剩多少空间" ✓
  ② `MemoryHistoryRecorder`  —— device 分配历史（窗口 + 快照 ✓）
      用途（原注释）: "回答是谁在每步做上百次 cudaMalloc" ✓
      （快照由 `tools/prof/mem_top.py` 聚合成"分配点排行榜" ✓）

★ 本模块是**纯观测** ✓:
  · **不**参与训练语义 ✗（不改 loss / 不改梯度 / 不 step ✓）
  · 默认**全关** ✓（`--profile-steps 0` / `--profile-memory-steps 0` ✓）
  · 只在窗口内开销 ✓（profiler 会拖慢被 profile 的步 ✓ ⇒ 默认关 ✓ 这是刻意的 ✓）

★ 调用约定（与旧内联代码**逐字一致** ✓）:
  · `maybe_start(batch_idx)` 在**每步开头**调用 ✓（到点自动 `__enter__` ✓）
  · `tick()` 在**完成 optimizer.step() 之后**调用 ✓
    ⚠ 旧代码把这两段放在 `if _finite:` 分支**内部** ✓ ⇒ 接线时**必须保持** ✗
      （即"非有限步不计入 profile 窗口" ✓ 语义不变 ✓）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

__all__ = ["StepProfiler", "MemoryHistoryRecorder", "Diagnostics",
           "WindowAverages", "average_window", "format_step_log"]


class StepProfiler:
    """算子级 profile ✓（窗口 N 步 ✓ 跳过 warmup ✓ 报表文本逐字保留 ✓）。"""

    def __init__(self, args, logger=None):
        self.logger = logger
        self.n = int(getattr(args, "profile_steps", 0) or 0)
        self.skip = int(getattr(args, "profile_skip", 20) or 20)
        self._obj = None
        self._cnt = 0

    @property
    def active(self) -> bool:
        return self._obj is not None

    def maybe_start(self, batch_idx: int) -> None:
        """窗口起点 ✓（逐字等价于旧内联 ✓）。"""
        if self.n and self._obj is None and batch_idx >= self.skip:
            self._obj = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False)
            self._obj.__enter__()
            self._cnt = 0
            if self.logger is not None:
                self.logger.info(f"[profile] 开始算子级 profile: 步 {batch_idx} 起 {self.n} 步 "
                                 f"(warmup {self.skip} 步已跳过, torch={torch.__version__})")

    def tick(self) -> None:
        """窗口终点 ✓（到点打印报表 ✓ 文本与旧实现**逐字一致** ✓）。"""
        if self._obj is None:
            return
        self._cnt += 1
        if self._cnt < self.n:
            return
        try:
            self._obj.__exit__(None, None, None)
            _avg = self._obj.key_averages()
            _cuda = sum(float(getattr(o, "self_cuda_time_total", 0.0))
                        for o in _avg) / 1e3 / max(self._cnt, 1)
            _cpu = sum(float(getattr(o, "self_cpu_time_total", 0.0))
                       for o in _avg) / 1e3 / max(self._cnt, 1)
            if self.logger is not None:
                self.logger.info(
                    f"[profile] N={self._cnt} 步 | 排他 CUDA 忙 = {_cuda:.1f} ms/步 | "
                    f"排他 CPU = {_cpu:.1f} ms/步 | "
                    f"（GPU 占空比 ≈ {_cuda:.0f}ms / 实测步时；两者差 = 启动/等待开销）")
                self.logger.info("[profile] 逐算子 CUDA 时间 TOP20:\n"
                                 + _avg.table(sort_by="self_cuda_time_total", row_limit=20))
                self.logger.info("[profile] 逐算子 CPU 时间 TOP15:\n"
                                 + _avg.table(sort_by="self_cpu_time_total", row_limit=15))
        except Exception as _pe:                                        # noqa: BLE001
            if self.logger is not None:
                self.logger.warning(f"[profile] 打印失败: {_pe}")
        self._obj = None

    def describe(self) -> str:
        return f"profile_steps={self.n} skip={self.skip}"


class MemoryHistoryRecorder:
    """device 分配历史 ✓（窗口 N 步 ✓ 快照写到 /tmp ✓ 与旧实现逐字一致 ✓）。"""

    def __init__(self, args, logger=None, snapshot_dir: str = "/tmp"):
        self.logger = logger
        self.n = int(getattr(args, "profile_memory_steps", 0) or 0)
        self.skip = int(getattr(args, "profile_memory_skip", 10) or 10)
        self.snapshot_dir = snapshot_dir
        self.on = False
        self._cnt = 0

    def maybe_start(self, batch_idx: int) -> None:
        if self.n and not self.on and batch_idx >= self.skip:
            import torch.cuda.memory as _tcm
            _tcm._record_memory_history(max_entries=300000)
            self.on = True
            self._cnt = 0
            if self.logger is not None:
                self.logger.info(f"[profile-mem] 开始记录 device 分配历史: "
                                 f"步 {batch_idx} 起 {self.n} 步")

    def tick(self) -> None:
        if not self.on:
            return
        self._cnt += 1
        if self._cnt < self.n:
            return
        import torch.cuda.memory as _tcm
        _snap = os.path.join(self.snapshot_dir, f"mem_snapshot_{os.getpid()}.pickle")
        try:
            _tcm._dump_snapshot(_snap)
            if self.logger is not None:
                self.logger.info(f"[profile-mem] 快照已写 {_snap} "
                                 f"(用 tools/prof/mem_top.py 解析)")
        except Exception as _me:                                        # noqa: BLE001
            if self.logger is not None:
                self.logger.warning(f"[profile-mem] 快照失败: {_me}")
        try:
            _tcm._record_memory_history(enabled=None)
        except Exception:                                               # noqa: BLE001
            pass
        self.on = False

    def describe(self) -> str:
        return f"profile_memory_steps={self.n} skip={self.skip}"


class Diagnostics:
    """两件观测工具的组合 ✓（`maybe_start` 在步首 ✓ `tick` 在 step 之后 ✓）。"""

    def __init__(self, args, logger=None):
        self.prof = StepProfiler(args, logger)
        self.mem = MemoryHistoryRecorder(args, logger)
        self.logger = logger

    def maybe_start(self, batch_idx: int) -> None:
        self.prof.maybe_start(batch_idx)
        self.mem.maybe_start(batch_idx)

    def tick(self) -> None:
        """⚠ 顺序与旧代码一致 ✓: **先** profiler ✓ **再** 分配历史 ✓"""
        self.prof.tick()
        self.mem.tick()

    @property
    def enabled(self) -> bool:
        return bool(self.prof.n or self.mem.n)

    def describe(self) -> str:
        if not self.enabled:
            return "diagnostics=off"
        return f"diagnostics: {self.prof.describe()} | {self.mem.describe()}"


# ══════════════════════════ 窗口均值汇总（纯计算 ✓） ══════════════════════════
@dataclass(frozen=True)
class WindowAverages:
    """一个日志窗口的**标量化**结果 ✓（纯数据 ✓ 无副作用 ✓）。

    ⚠ 与原内联代码的关系（**逐字等价** ✓）:
      · 累加器是 **float64 GPU 张量** ✓ ⇒ 除法在 GPU 上 ✓ **只在这一次**取标量 ✓
        （原注释: "累加器已是 GPU 张量 -> 除法仍在 GPU；**只在这一次**取标量" ✓）
      · `divisor = max(log_steps, 1)` ✓（防"整个窗口都因非有限 loss 被跳过"时除零 ✓）
      · `world_size > 1` 时先 `all_reduce(SUM)` 再 `item()/world_size` ✓
      · 12ch 分组 loss 从累加的 `(C,)` 向量拆出 image/canny/skel ✓
    """
    loss: float
    diff: float
    repa: float
    std_mid: float
    skel: float
    c12_img: float
    c12_canny: float
    c12_skel: float
    steps: int
    seconds: float
    steps_per_sec: float


def average_window(running, divisor: int, world_size: int, *,
                   steps: int = 0, seconds: float = 0.0,
                   aux_dirs: Sequence[str] = (), acc_c12=None,
                   all_reduce: Optional[Callable] = None) -> WindowAverages:
    """把窗口累加器折算成**标量均值** ✓ —— 逐字等价于 `train.py` 里的内联版本 ✓。

    参数:
      running    : dict ✓ 键 = `loss/diff/repa/std_mid/skel` ✓（float64 张量 ✓）
      divisor    : 窗口内**有效**步数 ✓（调用方应传 `max(log_steps, 1)` ✓）
      world_size : 进程数 ✓（>1 时走 all_reduce ✓）
      all_reduce : 可注入的归约回调 ✓（默认 `torch.distributed.all_reduce` ✓）
                   ⇒ 让本函数**可单测** ✓ 不硬依赖分布式初始化 ✓
      acc_c12    : 12ch 逐通道 MSE 累加向量 ✓（None ⇒ 分组全 0 ✓）
      aux_dirs   : aux 组名 ✓（用于识别 canny / skel ✓）
    """
    d = float(max(int(divisor), 1))
    _ar = all_reduce
    if _ar is None:
        import torch.distributed as _dist
        _ar = _dist.all_reduce
    out = {}
    for key in ("loss", "diff", "repa", "std_mid"):
        v = running[key] / d
        if world_size > 1:
            _ar(v, op=torch.distributed.ReduceOp.SUM)
            v = v.item() / world_size
        else:
            v = v.item()
        out[key] = float(v)
    skel = float((running["skel"] / d).item())
    c12i = c12c = c12s = 0.0
    if acc_c12 is not None:
        _cm = acc_c12 / d
        c12i = float(_cm[:4].mean())
        for _gi, _nm in enumerate(aux_dirs):
            _gm = float(_cm[4 + 4 * _gi: 8 + 4 * _gi].mean())
            if "canny" in _nm:
                c12c = _gm
            elif "skel" in _nm:
                c12s = _gm
    _secs = float(seconds)
    return WindowAverages(
        loss=out["loss"], diff=out["diff"], repa=out["repa"], std_mid=out["std_mid"],
        skel=skel, c12_img=c12i, c12_canny=c12c, c12_skel=c12s,
        steps=int(steps), seconds=_secs,
        steps_per_sec=(float(steps) / max(_secs, 1e-9)),
    )

# ══════════════════════════ 单步日志文本（纯格式化 ✓） ══════════════════════════
def format_step_log(*, step: int, avg_d: float, avg_r: float, avg_std_mid: float,
                    avg_skel: float, steps_per_sec: float, w_repa: float,
                    w_std_mid: float = 0.0, w_deform_skel: float = 0.0,
                    deform_loss: float = 0.0, deform_off: float = 0.0,
                    w_latent_skel: float = 0.0, vae_kl=None, w_vae_kl: float = 1e-5,
                    train_vae_enc: bool = False, vae_lr=None, lr: float = 0.0,
                    ema_decay=None, mem=None) -> str:
    """构造单步训练日志行 ✓（**纯格式化** ✓ 无副作用 ✓ 便于单测 ✓）。

    ⚠ 搬移要点 ✗: 原代码用 `globals().get('_DEFORM_EMA')` 读形变损失的 EMA ✗
      ⇒ 函数化后 `globals()` 变成**本模块**的 ✗ ⇒ 会读到空 ⇒ 形变项恒为 0 ✗✗
      ⇒ 因此改成**显式参数** `deform_loss` / `deform_off` ✓（由调用方传入 ✓）

    段落的**条件**与**格式**与旧实现逐字一致 ✓:
      · StdMid   仅在 `w_std_mid > 0` ✓
      · Deform   仅在 `w_deform_skel > 0` ✓
      · SkelStruct 仅在 `w_latent_skel > 0` ✓
      · VAE-KL   仅在 `train_vae_enc 且 vae_kl is not None` ✓
      · VAE-LR   仅在 `train_vae_enc 且组数 > 2` ✓（此处由 `vae_lr is not None` 表达 ✓）
      · EMA      仅在有 ema_decay 时 ✓
    """
    if mem is None:
        mem = (torch.cuda.memory_reserved() / 1024 ** 3,
               torch.cuda.max_memory_reserved() / 1024 ** 3)
    ema_log = (f"EMA: {ema_decay:.6f} | " if ema_decay is not None else "")
    return (
        f"(step={step:07d}) Diff: {avg_d:.4f} | "
        f"REPA(w={w_repa:.2f}): {avg_r:.4f} | "
        + (f"StdMid(w={w_std_mid:.3f}): {avg_std_mid:.4f} | " if w_std_mid > 0 else "")
        + (f"Deform(w={w_deform_skel:.2f}): {deform_loss:.4f} off={deform_off:.4f} | "
           if w_deform_skel > 0 else "")
        + (f"SkelStruct(w={w_latent_skel:.3f}): {avg_skel:.4f} | "
           if w_latent_skel > 0 else "")
        + (f"VAE-KL(w={w_vae_kl:.1e}): {float(vae_kl):.4f} | "
           if train_vae_enc and vae_kl is not None else "")
        + (f"VAE-LR: {vae_lr:.2e} | " if train_vae_enc and vae_lr is not None else "")
        + f"LR: {lr:.2e} | {ema_log}"
        f"Steps/Sec: {steps_per_sec:.2f} | "
        f"Mem: {mem[0]:.2f}G/{mem[1]:.2f}G"
    )
