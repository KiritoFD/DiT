# -*- coding: utf-8 -*-
"""eval_callback.py — **训练侧的评测回调**（2026-10-13 [审查 §4.1] 从 src/eval/in_mem_eval.py 迁出）

审查迁移表: "in_mem_eval.py:1327-1392 -> train/eval_callback.py：
            **训练器决定何时评测，评测器不决定训练调度**" ✓

本模块只回答"这一步要不要评测 / 用哪份权重 / 结果怎么落盘", 真正的评测在 src/eval/in_mem_eval.py ✓:
  · 只有 inline and --in-mem-eval and is_eval_step 才跑 ✓
  · 优先常驻 EMA（与 ckpt 落盘的是同一份 ✓），没有则用裸模型（DDP 时取 .module ✓）
  · 同步写 eval_auto_<step>.json —— train.py 里 EarlyStopper 的**主数据源** ✓
    （历史坑: 它只认 json, 而现代路径只写 CSV ⇒ 早停静默永不触发, 不报错、只是永远不停 ✗）
  · **失败只 warning 不中断训练**（评测不该把训练带崩 ✓），但打完整 traceback ✓

⚠ 依赖方向: 本模块 -> src.eval.in_mem_eval ✓（单向 ✓）。
  因此 in_mem_eval 侧**不能**放模块级 shim（会成环 ✗）⇒ 用**惰性**再导出 ✓。
"""
from __future__ import annotations

import os
import time

from src.eval.in_mem_eval import run_in_mem_eval

__all__ = ["maybe_run_in_training"]


def _fmt_eval(name, v):
    """汇总行。缺指标时打印 NA, 不让格式化把整次评测打成失败。"""
    if not isinstance(v, dict):
        return f"{name} ssim={v:.4f}"

    def n(x, spec):
        return "NA" if x is None else format(x, spec)

    return (f"{name} ssim={n(v.get('ssim'), '.4f')} "
            f"ink={n(v.get('ink_ssim'), '.4f')} "
            f"frag={n(v.get('frag'), '.3f')} "
            f"hole={n(v.get('hole'), '.3f')}")


def maybe_run_in_training(args, ema_model, model, train_steps, device,
                          checkpoint_dir, logger, is_eval_step, inline=True):
    """训练循环里的在训评测入口 —— 把"是否该跑 / 用哪份权重 / 计时 / 报错"都收在这里。

    train.py 侧只需一行调用：

        from src.eval.in_mem_eval import maybe_run_in_training
        maybe_run_in_training(args, ema_model, model, train_steps, device,
                              checkpoint_dir, logger, is_eval_step=_save_ckpt,
                              inline=_EVAL_INLINE)

    行为（与原内联实现完全一致）：
      * 只有 `inline and --in-mem-eval and is_eval_step` 才跑
      * 优先用常驻 EMA 权重（与 ckpt 落盘的是同一份），没有 EMA 就用裸模型
      * 显存：训练 ~17G + eval ~2G ≈ 19.5G < 24G
      * **失败只 warning 不中断训练**（评测不该把训练带崩），但会打完整 traceback
    返回 results dict，未跑则返回 None。
    """
    if not (inline and getattr(args, "in_mem_eval", False) and is_eval_step):
        return None
    _em = (ema_model if ema_model is not None
           else (model.module if hasattr(model, "module") else model))
    _em.eval()
    t0 = time.time()
    try:
        res = run_in_mem_eval(
            _em, args, train_steps, device,
            # ★ 2026-10-04: 评测产物必须落在**本次 run 自己的**目录 (train.py 已把
            #   args.eval_dir 解析为 run 目录), 否则共用目录下 (step,set) 去重会
            #   把整轮评测静默跳过。兜底: 由 checkpoint_dir 反推 run 目录。
            results_dir=str(getattr(args, "eval_dir", "") or "").strip()
            or os.path.dirname(os.path.dirname(checkpoint_dir)))
        # ★ 2026-10-03: 同步落 eval_auto_<step>.json 到 ckpt 目录 —— train.py 里
        #   EarlyStopper 的**主数据源**就是这个文件名 (它以前只认 json, 而现代路径
        #   只写 CSV -> 早停静默永不触发, 不报错、只是永远不停)。取 n 最大的 set
        #   (= 最可信) 拍平成扁平指标 (含 skel_iou / lpips, 供 iou_lpips 判据)。
        try:
            import json as _json
            _best = None
            for _s, _v in (res or {}).items():
                if not isinstance(_v, dict):
                    continue
                if _best is None or int(_v.get("n") or 0) > int(
                        res[_best].get("n") or 0):
                    _best = _s
            if _best is not None:
                _flat = {k: v for k, v in res[_best].items()
                         if isinstance(v, (int, float))}
                _flat.update({"step": int(train_steps), "set": _best})
                with open(os.path.join(checkpoint_dir,
                                       f"eval_auto_{int(train_steps)}.json"),
                          "w", encoding="utf-8") as _f:
                    _json.dump(_flat, _f, ensure_ascii=False, indent=2)
                logger.info(
                    f"[in-mem-eval] 早停数据源已落盘 eval_auto_{int(train_steps)}.json "
                    f"(set={_best} n={_flat.get('n')} skel_iou={_flat.get('skel_iou')} "
                    f"lpips={_flat.get('lpips')} ssim={_flat.get('ssim')})")
        except Exception as _je:                              # noqa: BLE001
            logger.info(f"[in-mem-eval] ⚠ eval_auto json 落盘失败(不影响训练/评测): {_je!r}")
        logger.info(
            f"[in-mem-eval] step {train_steps} done in {time.time() - t0:.0f}s: "
            + " | ".join(_fmt_eval(k, v) for k, v in res.items()))
        return res
    except Exception as e:                                    # noqa: BLE001
        logger.warning(f"[in-mem-eval] step {train_steps} FAILED: {e}", exc_info=True)
        return None
