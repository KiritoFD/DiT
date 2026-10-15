# -*- coding: utf-8 -*-
"""checkpoint.py — **CheckpointManager**（2026-10-14 [重构 PR4] ✓）

审查 §4.2 原文:
  "尤其不要把所有功能都塞进一个通用 `Trainer` 类。我的目标是：`Trainer` 控制循环，
   `TrainStep` 完成一次更新，`ObjectiveComposer` 组合损失，**`CheckpointManager`
   负责持久化**，各自保持小而清晰。" ✓

★ 本模块是**薄门面** ✓ —— 实现仍在 `src/train/ckpt.py`（审查已认可的既有拆分 ✓:
  临时文件 + fsync + os.replace + SHA256 manifest + 异步写线程 ✓）。
  这里只做三件事 ✓:
    ① 把"存盘点的**状态**"（目录 / keep / strict / 上次存到第几步 ✓）收成对象 ✓
    ② 把"存盘 = 组装 + 入队 + 轮转"这个**两行序列**收成一个方法 ✓
       （调用方此前要连续写 `save_checkpoint(...)` 7 个参数 + `prune_checkpoints(...)` ✓）
    ③ 提供 `should_save()` / `describe()` ✓（让存盘条件与身份**显式** ✓）

★ 边界（审查原则 B ✓）: 本模块
  · **不**碰 optimizer / scheduler 的 step ✗
  · **不**决定何时评测 ✗（评测是评测器的事 ✓）
  · **不**改日志周期 ✗
  它只回答"该不该存 ✓ 存了第几步 ✓ 收尾是否成功 ✓"。
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """ckpt 持久化的**状态所有者** ✓（目录 / 轮转 / 严格模式 / 上次存盘点 ✓）。

    用法（与旧的两行调用**逐字等价** ✓）::

        ckpt = CheckpointManager(checkpoint_dir, keep=int(getattr(args, 'ckpt_keep', 0)),
                                 logger=logger)
        ...
        if rank == 0 and ckpt.should_save(train_steps, _epoch_steps):
            ckpt.save(model, opt, ema_model, scheduler, args, train_steps)
        ...
        if dist.get_rank() == 0:
            ckpt.finalize(timeout=600)
    """

    def __init__(self, checkpoint_dir: str, keep: int = 0, logger=None,
                 strict: Optional[bool] = None, enabled: bool = True):
        self.checkpoint_dir = str(checkpoint_dir)
        self.keep = int(keep or 0)
        self.logger = logger
        self.strict = strict                    # None ⇒ 由 ckpt.py 的默认（严格 ✓）决定 ✓
        self.enabled = bool(enabled)
        self._last_saved: Optional[int] = None
        self._n_saved = 0

    # ── 存盘条件（与旧写法逐字一致 ✓）──
    def should_save(self, step: int, every: int) -> bool:
        """`every > 0 且 step % every == 0` ✓ —— 逐字等价于旧的
        `_epoch_steps > 0 and train_steps % _epoch_steps == 0` ✓

        ⚠ 旧调用点还额外要求 `step > 0` ✓（避免开头 0 步存盘 ✓）——
           那一条**仍留在调用方** ✓（它是"训练语义"而不是"持久化语义" ✓）。
        """
        return int(every) > 0 and int(step) % int(every) == 0

    # ── 存盘（= 组装 + 入队 + 轮转 ✓）──
    def save(self, model, opt, ema_model=None, scheduler=None, args=None,
             step: int = 0) -> Optional[str]:
        """组装(同步搬 CPU ✓) + **异步**入队 ✓ + 轮转 ✓ ⇒ 返回 ckpt 路径 ✓。

        逐字等价于旧的::

            save_checkpoint(model, opt, ema_model, scheduler, args,
                            train_steps, checkpoint_dir, logger)
            prune_checkpoints(checkpoint_dir, int(getattr(args, 'ckpt_keep', 0)), logger)
        """
        if not self.enabled:
            return None
        from src.train.ckpt import prune_checkpoints, save_checkpoint
        path = save_checkpoint(model, opt, ema_model, scheduler, args,
                               step, self.checkpoint_dir, self.logger)
        prune_checkpoints(self.checkpoint_dir, self.keep, self.logger)
        self._last_saved = int(step)
        self._n_saved += 1
        return path

    # ── 收尾 ──
    def finalize(self, timeout: int = 600):
        """等全部落盘 + 停线程 + **严格模式**报错 ✓ ⇒ 返回 `(成功数, 错误列表)` ✓。

        逐字等价于旧的 `drain_ckpt(logger, timeout=600)` ✓
        （`strict=None` ⇒ 走 ckpt.py 的默认 = **严格** ✓ 失败会让进程非零退出 ✓ P1-3a ✓）
        """
        from src.train.ckpt import drain_ckpt
        return drain_ckpt(self.logger, timeout=timeout, strict=self.strict)

    # ── 状态读取 ──
    @property
    def last_saved(self) -> Optional[int]:
        """上次成功**入队**的步数 ✓（None = 还没存过 ✓）。"""
        return self._last_saved

    @property
    def n_saved(self) -> int:
        return self._n_saved

    def describe(self) -> str:
        """一行摘要 ✓（供启动日志 / manifest ✓ 让"存盘策略"成为**显式身份** ✓）。"""
        return (f"ckpt_dir={os.path.basename(self.checkpoint_dir.rstrip('/')) or self.checkpoint_dir} "
                f"keep={self.keep} strict={'default(严格)' if self.strict is None else self.strict} "
                f"enabled={self.enabled}")
