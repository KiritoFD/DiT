"""src.eval — 推理与评估层.

核心引擎在 ``src.eval.inference`` (统一采样/解码/指标);
其余模块 (auto_eval_*, in_process_*, eval_*) 是调用核心引擎的壳/daemon。
"""

__all__ = []