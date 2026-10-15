# -*- coding: utf-8 -*-
"""test_refactor_pr3.py —— 2026-10-14 [重构 PR3] **单一采样器 / 评测入口收敛** 的不变量守卫 ✓

审查 §4.3 原文: "目前 `src/eval/` 的 48 个 Python 文件中, 包含多代 GPU batch eval、
  CPU daemon、in-process eval、control eval 和自动评测入口。我建议最终**仅保留一套
  采样与评测语义**, 不同执行方式只是调度策略。" ✓
  "`gpu_batch_eval.py`、`gpu_batch_eval_v2.py` | **新采样器通过验收后退役**" ✓
  并给出正确删除流程（静态扫描 → config 扫描 → 动态 import → checkpoint 兼容 →
  迁移外部调用 → 移出运行时路径 → **一个发布周期后删除**）✓

本组测试把"退役决策 + 单一采样器"固化成**可回归的不变量** ✓:
  ① 已归档的两个 legacy GPU eval **必须不在运行时路径** ✓（也不得被 import ✗）
  ② 主评测入口**必须复用** `src/eval/sampler.py` ✓（不得再自实现采样循环 ✗）
  ③ **活跃入口绝不能被误归档** ✗（`auto_eval_cpu` / `cpu_eval_daemon` 等以**脚本**
     方式被配置调用 ✓ ⇒ 它们不是死代码 ✓）
"""
from __future__ import annotations

import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "src", "eval")

ARCHIVED = ("gpu_batch_eval.py", "gpu_batch_eval_v2.py")
# 以**脚本/子进程**方式被调用（审查流程里"外部工具调用"那一环 ✓）⇒ 绝不可归档 ✗
LIVE_BY_CLI = ("auto_eval_cpu.py", "cpu_eval_daemon.py", "cpu_eval_worker.py",
               "auto_eval_ctrl.py", "in_process_eval.py")
# 主评测入口 ⇒ 必须复用共享采样器 ✓
MUST_USE_SAMPLER = ("in_mem_eval.py", "batch_eval.py", "eval_ctrl_ckpt.py")


def test_archived_files_left_runtime_path():
    for f in ARCHIVED:
        assert not os.path.isfile(os.path.join(EVAL, f)), \
            f"{f} 仍在运行时路径 ✗（PR3: 已退役 ⇒ 应在 legacy/ ✓）"
        assert os.path.isfile(os.path.join(EVAL, "legacy", f)), \
            f"{f} 未归档 ✗（历史可复现性必须保留 ✓）"


def test_archived_files_have_retirement_note():
    for f in ARCHIVED:
        s = io.open(os.path.join(EVAL, "legacy", f), encoding="utf-8").read()
        assert "已退役" in s, f"{f} 缺少退役说明 ✗"
        assert "sampler.py" in s, f"{f} 必须指向替代实现 ✗"


def test_legacy_init_documents_removal_process():
    s = io.open(os.path.join(EVAL, "legacy", "__init__.py"), encoding="utf-8").read()
    assert "静态引用扫描" in s and "发布周期后删除" in s, \
        "legacy/__init__.py 必须记录审查的删除流程 ✓"
    assert "归档不等于删除" in s, "必须说明归档≠删除 ✓（历史可复现 ✓）"


def test_no_runtime_imports_of_archived_modules():
    """★ 旧路径必须**不可被 import** ✗（否则退役没有真正生效 ✗）"""
    import importlib
    for mod in ("src.eval.gpu_batch_eval", "src.eval.gpu_batch_eval_v2"):
        try:
            importlib.import_module(mod)
            raise AssertionError(f"{mod} 仍可导入 ✗（PR3 退役未生效 ✗）")
        except ModuleNotFoundError:
            pass
    # 归档路径仍可导入 ✓（历史实验可复现 ✓）
    importlib.import_module("src.eval.legacy.gpu_batch_eval")


def test_live_cli_entries_are_not_archived():
    """★ 防**过度归档** ✗: 以脚本方式被配置调用的入口必须留在运行时路径 ✓"""
    for f in LIVE_BY_CLI:
        assert os.path.isfile(os.path.join(EVAL, f)), \
            f"{f} 是**活跃**入口（脚本/子进程调用 ✓）⇒ 不得归档 ✗"
        assert not os.path.isfile(os.path.join(EVAL, "legacy", f)), \
            f"{f} 被误归档 ✗（PR3 明确要求先迁移外部调用 ✓）"


def test_main_entries_use_shared_sampler():
    """★ PR3 完成标准的一半 ✓: 主评测入口必须**复用**单一采样语义 ✓

    允许两种取得方式（都是同一份实现 ✓）:
      · 直接从 `src.eval.sampler` 导入 ✓
      · 经 `src.eval.inference` 的**兼容再导出** ✓（下一条测试锁定它确实是同一对象 ✓）
    """
    for f in MUST_USE_SAMPLER:
        p = os.path.join(EVAL, f)
        if not os.path.isfile(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        assert re.search(r"from\s+src\.eval\.(sampler|inference)\s+import", s), \
            f"{f} 未从 sampler/inference 导入采样函数 ✓（审查: 单一采样语义 ✓）"
        assert "sample_latents" in s, f"{f} 未调用 sample_latents ✓"
        # ★ 不得再出现**遗留 DDPM/DDIM** 采样标记 ✗（这正是两代实现的分界 ✓）
        for bad in ("ddim_sample_loop", "create_diffusion("):
            assert bad not in s, f"{f} 仍含遗留 DDPM 采样 `{bad}` ✗（PR3: 收敛到 FM 主线 ✗）"


def test_inference_reexports_same_objects():
    """★ 兼容层必须是**同一函数对象** ✓（否则"单一语义"名存实亡 ✗）"""
    from src.eval import inference, sampler
    for name in ("sample_latents", "sample_latents_self_cond", "build_diffusion"):
        assert hasattr(sampler, name), f"sampler 缺 {name} ✗"
        assert getattr(inference, name, None) is getattr(sampler, name), \
            f"inference.{name} 与 sampler.{name} **不是同一对象** ✗（存在重复实现 ✗）"


def test_sampler_module_is_the_single_semantics():
    s = io.open(os.path.join(EVAL, "sampler.py"), encoding="utf-8").read()
    assert "def sample_latents(" in s, "sampler.py 必须提供 sample_latents ✓"
    assert "def build_diffusion(" in s, "sampler.py 必须提供 build_diffusion ✓"
    # 共享实现不得依赖 legacy ✗
    assert "legacy" not in s, "sampler.py 不得依赖 legacy ✗"


def test_eval_dir_line_count_did_not_grow():
    """粗粒度守卫 ✓: 退役应让 src/eval 顶层行数**不增长** ✓（PR7 会继续压 ✓）"""
    n = 0
    for f in os.listdir(EVAL):
        if f.endswith(".py"):
            n += len(io.open(os.path.join(EVAL, f), encoding="utf-8").read().split("\n"))
    assert n < 10500, f"src/eval 顶层行数异常增长: {n} ✗"
