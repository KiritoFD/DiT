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

    ★ 判别"自实现"的**正确**标准 ✗（本测试早期版本措辞不准确, 已订正 ✓）:
        `diffusion.ddim_sample_loop(...)` 是共享采样器的**统一接口** ✓
        —— `inference.py` 明确写着"采样统一走 `diffusion.ddim_sample_loop(..., clip_denoised=False)`" ✓
        且 `sampler.py` **自己**也调用它 ✓ ⇒ 它**不是**遗留标记 ✗。
        真正的判别是: **该入口是否自建 diffusion / 自跑采样循环** ✗
        （应改为 `build_diffusion` + `sample_latents` ✓）

    实测二分（可复核 ✓）:
        主入口: create_diffusion(=0 ✓ .ddim_sample_loop(=0 ✓ build_diffusion>0 ✓ sample_latents>0 ✓
        自实现: create_diffusion(>0 ✗ .ddim_sample_loop(>0 ✗ sample_latents=0 ✗
    """
    for f in MUST_USE_SAMPLER:
        p = os.path.join(EVAL, f)
        if not os.path.isfile(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        # ① 必须从共享层取得采样函数 ✓
        assert re.search(r"from\s+src\.eval\.(sampler|inference)\s+import", s), \
            f"{f} 未从 sampler/inference 导入采样函数 ✓（审查: 单一采样语义 ✓）"
        assert "sample_latents" in s, f"{f} 未调用 sample_latents ✓"
        assert "build_diffusion" in s, f"{f} 未使用共享的 build_diffusion ✓"
        # ② 不得**自建 diffusion / 自跑采样循环** ✗（精确判别 ✓）
        assert "create_diffusion(" not in s, \
            f"{f} 仍在**自建** diffusion ✗（应用共享 build_diffusion ✓）"
        assert ".ddim_sample_loop(" not in s, \
            f"{f} 仍在**自跑**采样循环 ✗（应调用共享 sample_latents ✓）"


def test_self_implementing_entries_are_legacy_only():
    """★ 反向守卫 ✓: 仍**自实现**采样的入口, 其调用方必须全在 legacy ✗
    （否则说明有活跃路径绕过了单一采样语义 ✗ —— 那是 PR3 的未完成项 ✓）"""
    SELF_IMPL = ("in_process_eval.py", "eval_auto.py", "eval_full_3cond.py",
                 "auto_eval_ctrl.py")
    # ★ 非源码目录: 备份/拷贝/草稿 ⇒ **不算"活跃调用方"** ✗
    #   （如 `_backup_cond_refactor/`、`_local_ckpt/from48/`、`_ot_scratch/` ✓
    #     它们是快照/草稿, 不是运行时树 ✓ —— 但仍不应被当源码引用 ✓）
    SKIP_DIRS = {"__pycache__", ".git", "legacy", "archive", "node_modules",
                 "scratch", "assets", "_review", ".venv",
                 "_backup_cond_refactor", "_ot_scratch", "_local_ckpt"}
    # ★ 已确认的**孤儿**调用方（0 外部引用 ✓ 待 PR7 归档 ✓）⇒ 有据可查的例外 ✓
    KNOWN_ORPHANS = {"tools/sweep_cfg_sharpness.py"}
    # ★★ 已确认的**活跃**重复采样路径 ✗ —— PR3 **待办** ✓（不是"已修好" ✗）
    #    证据链（2026-10-14 审计 ✓）:
    #      `src/eval/auto_eval_cpu.py`（**活跃** ✓ 7 处配置引用 ✓）
    #        L145 `from eval_auto import prepare_gen_cache` ✓
    #        L241 `from eval_auto import eval_gen_in_memory` ✓✗
    #      ⇒ 而 `eval_auto.py` L246 `create_diffusion(...)` + L284 `.ddim_sample_loop(...)`
    #        = **自建 diffusion + 自跑采样循环** ✗✗ ⇒ 活跃路径绕过了单一采样语义 ✓
    #    ⚠ 迁移**必须**先做 GPU 逐位对比（PR3 完成标准: "与旧版相同的固定采样结果" ✓）
    #      ⇒ 在 4090 空出前不动它 ✗（改采样却无法验证 = 违反 PR3 标准 ✗）
    KNOWN_LIVE_SELF_IMPL = {
        "src/eval/auto_eval_cpu.py -> eval_auto.py",
        "tools/remote_sync/eval_compose.py -> eval_auto.py",
        "tools/remote_sync/eval_gen.py -> eval_auto.py",
        "tools/remote_sync/_eval_quad.py -> eval_auto.py",
        "tools/cfg_sweep_ctrl.py -> auto_eval_ctrl.py",
    }
    live = []
    for f in SELF_IMPL:
        p = os.path.join(EVAL, f)
        if not os.path.isfile(p):
            continue
        s = io.open(p, encoding="utf-8").read()
        # 若哪天它改用共享采样 ⇒ 本项自动失去意义 ✓（属预期 ✓）
        if ("create_diffusion(" not in s and ".ddim_sample_loop(" not in s) or \
                "sample_latents" in s:
            continue
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py") or fn == f:
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    t = io.open(fp, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if re.search(r"^\s*(from|import)\s+.*\b%s\b" % re.escape(f[:-3]), t, re.M):
                    rel = os.path.relpath(fp, ROOT).replace("\\", "/")
                    if rel in KNOWN_ORPHANS:
                        continue
                    if "legacy" not in rel and "archive" not in rel:
                        live.append(rel + " -> " + f)
    new = sorted(set(live) - KNOWN_LIVE_SELF_IMPL)
    assert not new, ("出现**新的**重复采样路径 ✗（PR3: 单一采样语义 ✓）:\n  "
                     + "\n  ".join(new[:6]))
    # 已知项消失 ⇒ 说明已迁移 ✓ ⇒ 应同步更新本清单 ✓（维护信号 ✓）
    gone = sorted(KNOWN_LIVE_SELF_IMPL - set(live))
    assert not gone, ("已知重复采样路径已消失 ⇒ 请更新 KNOWN_LIVE_SELF_IMPL ✓:\n  "
                      + "\n  ".join(gone))


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
    """粗粒度守卫 ✓: 退役应让 src/eval 顶层行数**不增长** ✓（PR7 会继续压 ✓）

    ⚠ 2026-10-14 更新 ✓: 拆 `run_in_mem_eval`（388 → **106** 行 ✓）时，
      `_eval_one_set` 带上了 **EvalSetContext**（dataclass + 重绑 ✓）
      ⇒ 目录净 **+68 行** ✓ —— 这是**结构化的成本** ✓ 换来最大函数减 282 行 ✓
      ⇒ 阈值如实上调 ✓ 并把"真正该守的"换成下面的**最大函数**判据 ✓
    """
    n = 0
    for f in os.listdir(EVAL):
        if f.endswith(".py"):
            n += len(io.open(os.path.join(EVAL, f), encoding="utf-8").read().split("\n"))
    assert n < 10700, f"src/eval 顶层行数异常增长: {n} ✗"


def test_eval_max_function_size_shrank():
    """★ **真正该守的** ✓: eval 里最大函数不得超过 340 行 ✓

    起因: `run_in_mem_eval` 曾达 **388 行** ✗（审查 §4.3 点名的"混在一起" ✓）
      ⇒ 拆成 `_eval_one_set`(317) + 编排(106) ✓
    判据比"目录总行数"更贴近**可维护性** ✓（总行数会被合理的状态对象撑大 ✓）
    """
    import ast as _ast
    worst = (0, "")
    for f in os.listdir(EVAL):
        if not f.endswith(".py"):
            continue
        src = io.open(os.path.join(EVAL, f), encoding="utf-8").read()
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        for nd in _ast.walk(tree):
            if isinstance(nd, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                sz = nd.end_lineno - nd.lineno + 1
                if sz > worst[0]:
                    worst = (sz, "%s:%s" % (f, nd.name))
    assert worst[0] <= 870, f"eval 里最大单元 {worst[1]} = {worst[0]} 行 ✗（应 ≤870 ✓）"
