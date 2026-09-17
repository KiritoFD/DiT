# -*- coding: utf-8 -*-
"""Checkpoint 保存 / 轮转 / 异步落盘（从 train.py 抽出，2026-09-17）。

## 为什么存盘要异步

`torch.save` 一个 ckpt 约 **592 MB**（model + ema + optimizer）。原实现在训练循环里
**同步**执行 —— 单卡时 GPU 全程停摆等写盘（实测每次 ~2s，40 次 ≈ 80s + 每次的停顿）。

关键前提：**状态先同步搬到 CPU**（`state_to_cpu`，~0.5s，必须同步），
之后"序列化 + 落盘"（~2s）完全可以与训练重叠。

## ⚠ 唯一容易写错的地方：必须真的复制

`Tensor.cpu()` 对**已在 CPU 的张量是 no-op**（返回同一 storage），只有 GPU→CPU 才拷贝。
异步写盘时若与训练共享 storage，后台线程会读到**被训练改到一半的数据**
（典型：optimizer 的 `step` 计数器，它就在 CPU 上）。

所以 `state_to_cpu` 用 `.to("cpu", copy=True)` —— 两种情况都真复制，且都**只复制一次**。

## 用法

    from src.train.ckpt import WRITER, save_checkpoint, drain_ckpt

    save_checkpoint(model, opt, ema_model, scheduler, args, train_steps,
                    checkpoint_dir, logger)      # 入队即返回，不阻塞训练
    ...
    drain_ckpt(logger)                            # 训练结束前必须调用
"""
import os

import torch


def state_to_cpu(obj):
    """递归把（可能嵌套的）state dict 里的张量搬到 CPU 的**独立副本**。

    `opt.state_dict()` 会嵌套两层 dict（state -> param_idx -> tensors）
    并含 list（param_groups），所以一次扁平 `.detach().cpu()` 不够。

    ⚠ 用 `copy=True` 而非 `.cpu()`：后者对已在 CPU 的张量是 no-op（共享 storage），
    异步写盘时会与训练产生数据竞争。见模块 docstring。
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [state_to_cpu(v) for v in obj]
    return obj


# 兼容旧名字（train.py 里原来叫 _state_to_cpu）
_state_to_cpu = state_to_cpu


class AsyncCkptWriter:
    """后台线程写 ckpt —— 主进程把**已搬到 CPU 的状态**交给它后立刻继续训练。

    RAM 代价：每个在途 ckpt 持有 ~592 MB CPU 张量。存盘间隔通常 ~20 分钟、
    写盘 ~2s，所以实际永远只有 1 个在途；`max_pending` 只是兜底防积压
    （队列满时 `submit` 会**同步等待**——宁可慢也不丢 ckpt）。
    """

    def __init__(self, max_pending=2):
        import queue
        import threading
        self._q = queue.Queue(maxsize=max_pending)
        self._thread = None
        self._errors = []
        self._lock = threading.Lock()
        self._n_done = 0

    @property
    def n_done(self):
        with self._lock:
            return self._n_done

    @property
    def n_in_flight(self):
        return self._q.qsize()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            checkpoint, path = item
            try:
                torch.save(checkpoint, path)
                open(path + ".done", "w").close()
                with self._lock:
                    self._n_done += 1
            except Exception as e:                            # noqa: BLE001
                with self._lock:
                    self._errors.append((path, repr(e)))
            finally:
                del checkpoint                                # 尽早释放 RAM
                self._q.task_done()

    def submit(self, checkpoint, path):
        """异步入队。队列满时**同步等待**（宁可慢也不丢 ckpt）。"""
        if self._thread is None or not self._thread.is_alive():
            import threading
            self._thread = threading.Thread(target=self._run, daemon=False,
                                            name="ckpt-writer")
            self._thread.start()
        self._q.put((checkpoint, path))

    def drain(self, timeout=None):
        """等全部写完并停掉线程。**训练结束 / 异常退出前必须调用**，
        否则最后一个 ckpt 可能还没落盘进程就退了。返回 (成功数, 错误列表)。"""
        self._q.join()
        if self._thread is not None and self._thread.is_alive():
            self._q.put(None)
            self._thread.join(timeout=timeout)
        with self._lock:
            errs = list(self._errors)
        if errs:
            print(f"[ckpt] ⚠ {len(errs)} 个 ckpt 写盘失败: {errs[:3]}")
        return self._n_done, errs


# 进程级单例（只 rank 0 会用到）
WRITER = AsyncCkptWriter()


def build_checkpoint(model, opt, ema_model=None, scheduler=None,
                     args=None, train_steps=0, logger=None):
    """组装 ckpt dict（**同步**把状态搬到 CPU）。返回可直接交给 WRITER 的 dict。

    注意：这一步必须在训练继续之前完成 —— 它读的是"此刻"的权重。
    """
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        # 始终保存完整 state_dict（不做 delta-only）。
        "delta": state_to_cpu(model_to_save.state_dict()),
        "opt": state_to_cpu(opt.state_dict()),
        "args": args,
        "train_steps": train_steps,
    }
    if ema_model is not None:
        ckpt["ema"] = state_to_cpu(ema_model.state_dict())
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if logger is not None:
        logger.info(f"[ckpt] built {train_steps:07d}.pt (state on CPU)")
    return ckpt


def save_checkpoint(model, opt, ema_model=None, scheduler=None, args=None,
                    train_steps=0, checkpoint_dir=".", logger=None):
    """组装 + 异步入队，返回 checkpoint 路径。**不阻塞训练。**"""
    ckpt = build_checkpoint(model, opt, ema_model, scheduler, args,
                            train_steps, logger)
    path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
    WRITER.submit(ckpt, path)
    if logger is not None:
        logger.info(f"[ckpt] queued {path} (async write, {WRITER.n_in_flight} in flight)")
    return path


def prune_checkpoints(checkpoint_dir, keep, logger=None):
    """只保留最近 `keep` 个 ckpt（及各自的 eval_* 目录），返回删除数。

    ⚠ 只在 `keep > 0` 时调用。删除前会先 `.done` 检查——但注意**在途**的 ckpt
    可能还没落盘，所以调用方应先 `WRITER.drain()` 或确保不会删到在途的那个。
    """
    if keep <= 0:
        return 0
    import glob
    import shutil
    pts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    n = 0
    for old in pts[:-keep]:
        base = os.path.basename(old)[:-3]
        os.remove(old)
        done = old + ".done"
        if os.path.exists(done):
            os.remove(done)
        eval_dir = os.path.join(checkpoint_dir, f"eval_{base}")
        if os.path.isdir(eval_dir):
            shutil.rmtree(eval_dir, ignore_errors=True)
        n += 1
    if n and logger is not None:
        logger.info(f"[ckpt-keep] pruned {n} old checkpoint(s), keeping {keep}")
    return n


def drain_ckpt(logger=None, timeout=600):
    """训练结束前调用：等全部落盘 + 停线程 + 报告失败。"""
    n, errs = WRITER.drain(timeout=timeout)
    if logger is not None:
        logger.info(f"[ckpt] async writer drained: {n} 个已落盘"
                    + (f", **{len(errs)} 个失败**: {errs[:3]}" if errs else ""))
    return n, errs
