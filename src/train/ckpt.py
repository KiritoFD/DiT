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
from typing import Optional

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
                # ★ 2026-10-13 [审查 P0]: **原子写盘** ✓ —— 原来直接 torch.save 到最终
                #   路径: 崩溃/被杀/并发读会看到**半文件** ✗, 而 ".done" 仅靠文件存在
                #   也不能证明权重完整 ✗。现在: 临时文件 -> fsync -> os.replace(原子 ✓)
                #   -> 写含 sha256/size/step 的完成 manifest ✓
                import hashlib
                import json as _json
                tmp = "%s.%d.tmp" % (path, os.getpid())
                # ⚠ fsync 必须在**可写**句柄上 ✗ —— 用只读句柄会 OSError(9) Bad file descriptor
                #   （本 session 实测踩过 ✓）。写法: 写完同一个句柄 flush+fsync 后再关 ✓
                with open(tmp, "wb") as _f:
                    torch.save(checkpoint, _f)
                    _f.flush()
                    os.fsync(_f.fileno())
                _h = hashlib.sha256()
                with open(tmp, "rb") as _f:
                    for _chunk in iter(lambda: _f.read(1 << 20), b""):
                        _h.update(_chunk)
                _size = os.path.getsize(tmp)
                os.replace(tmp, path)                       # 原子替换 ✓
                # ★ 2026-10-13 [审查]: 训练器写的是**顶层 "train_steps"** ✓, 而这里原来只读
                #   "step" ✗ ⇒ manifest 里的 step 恒为 null, 削弱实验溯源 ✗。
                #   现在按 train_steps > step > epoch 依次取（并保留原始来源名便于对账 ✓）
                _step, _step_src = None, None
                for _k in ("train_steps", "step", "global_step", "epoch"):
                    _v = checkpoint.get(_k) if isinstance(checkpoint, dict) else None
                    if _v is not None:
                        try:
                            _step, _step_src = int(_v), _k
                            break
                        except Exception:                       # noqa: BLE001
                            continue
                with open(path + ".done", "w", encoding="utf-8") as _f:
                    _json.dump({"sha256": _h.hexdigest(), "size": _size, "step": _step,
                                "step_source": _step_src, "atomic": True}, _f)
                    _f.flush(); os.fsync(_f.fileno())
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


def capture_rng() -> dict:
    """抓取**全部**随机源状态 ✓（2026-10-14 [重构 P1-2] ✓）。

    审查原文: "当前 checkpoint 包含模型、optimizer、EMA、scheduler、args、step，
    却没有完整保存 Python random / NumPy / CPU torch / CUDA torch 的 RNG" ✗
    ⇒ "恢复可以继续训练, 但**不保证逐步、逐样本可重放**" ✗ —— 而本项目经常比较
       5k–10k 的**微小增益** ✓ ⇒ 这种差别很重要 ✓。

    ⚠ 数据顺序: `LongEpoch` 采样器**每 epoch_steps 步 reset 一次** ✓ 且
    `train_steps % epoch_steps` 可**完全决定**它在 epoch 内的位置 ✓
    ⇒ 不需要额外存采样器状态 ✓（前提: `ckpt_every == epoch_steps` 的项目约定 ✓
       若二者不等, 恢复后 epoch 内相位会错位 ✗ —— 见 `rng_meta["epoch_aligned"]` ✓）。
    """
    import random as _random
    st: dict = {}
    try:
        st["python"] = _random.getstate()
    except Exception:                                             # noqa: BLE001
        pass
    try:
        import numpy as _np
        st["numpy"] = _np.random.get_state()
    except Exception:                                             # noqa: BLE001
        pass
    try:
        st["torch_cpu"] = torch.get_rng_state()
    except Exception:                                             # noqa: BLE001
        pass
    try:
        if torch.cuda.is_available():
            st["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:                                             # noqa: BLE001
        pass
    return st


def restore_rng(st: dict, logger=None) -> bool:
    """恢复 `capture_rng()` 的状态 ✓。返回是否**完整**恢复 ✓（缺项即 False ✓）。"""
    import random as _random
    if not st:
        if logger is not None:
            logger.warning("[ckpt] 该 ckpt **没有 RNG 状态** ✗ ⇒ 恢复后**不可逐样本重放** ✗")
        return False
    ok = True
    try:
        if "python" in st:
            _random.setstate(st["python"])
        else:
            ok = False
        if "numpy" in st:
            import numpy as _np
            _np.random.set_state(st["numpy"])
        else:
            ok = False
        if "torch_cpu" in st:
            torch.set_rng_state(st["torch_cpu"])
        else:
            ok = False
        if "torch_cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["torch_cuda"])
    except Exception as e:                                        # noqa: BLE001
        if logger is not None:
            logger.warning("[ckpt] RNG 恢复失败: %r ✗", e)
        return False
    if logger is not None:
        logger.info("[ckpt] RNG 状态已恢复 %s（python/numpy/torch_cpu/cuda ✓）"
                    % ("**完整** ✓" if ok else "**不完整** ✗"))
    return ok


def build_checkpoint(model, opt, ema_model=None, scheduler=None,
                     args=None, train_steps=0, logger=None):
    """组装 ckpt dict（**同步**把状态搬到 CPU）。返回可直接交给 WRITER 的 dict。

    注意：这一步必须在训练继续之前完成 —— 它读的是"此刻"的权重。

    ★ 2026-10-14 [重构 P1-2] 新增 `rng`（全部随机源 ✓）+ `rng_meta`（相位/对齐信息 ✓）:
        · `rng`         : capture_rng() ✓ ⇒ 支持**逐步可重放** ✓
        · `rng_meta`    : epoch_steps / epoch_aligned / resume_kind ✓
          `resume_kind` 明确区分两种恢复语义 ✓（审查要求 ✗ 不要靠一堆开关隐式组合 ✗）:
            "exact"      = 同一实验续训 ✓（要求 RNG + optimizer + scheduler 全恢复 ✓）
            "warm_start" = 只借权重 ✓（**故意**重建 optimizer 与随机序列 ✓）
    """
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        # 始终保存完整 state_dict（不做 delta-only）。
        "delta": state_to_cpu(model_to_save.state_dict()),
        "opt": state_to_cpu(opt.state_dict()),
        "args": args,
        "train_steps": train_steps,
        "rng": capture_rng(),
    }
    _ep = None
    try:
        _ep = int(getattr(args, "epoch_steps", 0) or 0) or None      # 0/缺失 ⇒ None ✓
    except Exception:                                             # noqa: BLE001
        _ep = None
    _ck = None
    try:
        _ck = int(getattr(args, "ckpt_every", 0) or 0) or None       # 0/缺失 ⇒ None ✓
    except Exception:                                             # noqa: BLE001
        _ck = None
    ckpt["rng_meta"] = {
        "epoch_steps": _ep,
        "ckpt_every": _ck,
        # 二者相等 ⇒ 恢复后 epoch 内相位可由 train_steps 完全重建 ✓（项目约定 ✓）
        "epoch_aligned": bool(_ep and _ck and _ep == _ck),
        "resume_kind": "exact",
    }
    if ema_model is not None:
        ckpt["ema"] = state_to_cpu(ema_model.state_dict())
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if logger is not None:
        logger.info(f"[ckpt] built {train_steps:07d}.pt (state on CPU, rng=✓, "
                    f"epoch_aligned={ckpt['rng_meta']['epoch_aligned']})")
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
    import hashlib as _hl
    import json as _js
    skipped = []
    for old in pts[:-keep]:
        base = os.path.basename(old)[:-3]
        # ★ 2026-10-13 [审查 v3]: 原实现**直接 os.remove** ✗（docstring 声称会查 .done ✗）
        #   半写入 / 在途 / 未完成校验的 ckpt 被删掉 => 评测可能引用到不存在的权重 ✗。
        #   现在: 只有 .done 存在、可解析、且 sha256 与 size 与实际文件**一致**才删 ✓,
        #   否则**保留并告警** ✓（宁可留几个旧 ckpt, 也不删错的 ✓）。
        _done = old + ".done"
        _ok = False
        try:
            m = _js.load(open(_done, encoding="utf-8"))
            _ok = (m.get("size") == os.path.getsize(old))
            if _ok and m.get("sha256"):
                h = _hl.sha256()
                with open(old, "rb") as f:
                    for c in iter(lambda: f.read(1 << 20), b""):
                        h.update(c)
                _ok = (h.hexdigest() == m["sha256"])
        except Exception:                                       # noqa: BLE001
            _ok = False
        if not _ok:
            skipped.append(base)
            continue
        os.remove(old)
        done = old + ".done"
        if os.path.exists(done):
            os.remove(done)
        if skipped and logger is not None and len(skipped) <= 3:
            logger.warning("[ckpt] prune 跳过 %d 个**未通过完成校验**的 ckpt（保留 ✓）: %s",
                           len(skipped), skipped[:3])
        eval_dir = os.path.join(checkpoint_dir, f"eval_{base}")
        if os.path.isdir(eval_dir):
            shutil.rmtree(eval_dir, ignore_errors=True)
        n += 1
    if n and logger is not None:
        logger.info(f"[ckpt-keep] pruned {n} old checkpoint(s), keeping {keep}")
    return n


def _ckpt_strict_default() -> bool:
    """严格模式开关 ✓（默认**开** ✓ —— 正式训练"ckpt 写失败"必须让进程失败 ✗）。

    可用 `DSH_CKPT_STRICT=0` 显式关闭（调试/临时环境 ✓）。
    ★ 2026-10-14 [重构 P1-3a] 审查指出: 旧实现只把异常记进 `_errors` ✗
    并在 drain 时**打印**一句 ✗ ⇒ 可能出现"训练跑完了, 但最终模型没可靠保存" ✗✗。
    """
    return os.environ.get("DSH_CKPT_STRICT", "1").strip().lower() not in ("0", "false", "no", "off")


def drain_ckpt(logger=None, timeout=600, strict: Optional[bool] = None):
    """训练结束前调用：等全部落盘 + 停线程 + 报告失败。

    ★ 2026-10-14 [重构 P1-3a] `strict=True`（默认 ✓）时:
        任一 ckpt 写盘/校验失败 ⇒ **抛 RuntimeError** ✗ ⇒ 调用方以**非零码**退出 ✓，
        并在日志里留下明确的失败状态 ✓（而不是"训练完成但模型没存" ✗✗）。
    """
    n, errs = WRITER.drain(timeout=timeout)
    _strict = _ckpt_strict_default() if strict is None else bool(strict)
    if logger is not None:
        logger.info(f"[ckpt] async writer drained: {n} 个已落盘"
                    + (f", **{len(errs)} 个失败**: {errs[:3]}" if errs else "")
                    + ("" if not errs else
                       (" | strict=ON ⇒ 将失败退出 ✗" if _strict else " | strict=OFF ⇒ 仅告警 ✗")))
    if errs and _strict:
        raise RuntimeError(
            f"[ckpt] {len(errs)} 个 checkpoint **写盘或校验失败** ✗（strict 模式 ✓）: "
            f"{errs[:3]} —— 拒绝以「成功」收尾 ✗（可用 DSH_CKPT_STRICT=0 临时关闭 ✓）")
    return n, errs
