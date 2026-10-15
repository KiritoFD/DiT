# -*- coding: utf-8 -*-
"""test_refactor_pr4h.py —— 2026-10-14 [重构 PR4] **telemetry（训练期诊断）** 的验收测试 ✓

审查 §3 目标结构含 `training/telemetry.py` ✓
审查 §6.3: "建议把性能监控分成三层: 吞吐 / GPU 时间 / 硬件利用" ✓
  —— 本模块落地其中**已有的两件观测工具** ✓（算子级 profile ✓ 分配历史 ✓）

本组测试锁定:
  ① 默认**全关** ✓（`--profile-steps 0` / `--profile-memory-steps 0` ✓）
  ② 窗口语义与旧内联**逐字一致** ✓（跳过 skip ✓ 窗口 n 步 ✓）
  ③ ★ 报表文本**逐字保留** ✓（便于按日志检索历史 ✓）
  ④ ★ 调用顺序: 先 profiler ✓ 再 分配历史 ✓
  ⑤ 纯观测边界 ✓（不碰 loss / 梯度 / optimizer ✓）
"""
from __future__ import annotations

import ast
import io
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import telemetry as TEL                                     # noqa: E402


def _read_train_src() -> str:
    """训练路径的源码 = `train.py` + `trainer.py` ✓（2026-10-14 循环已搬到 trainer.py ✓）。

    ⚠ 守卫不应**绑死文件位置** ✗ —— 它要守的是"**这段代码存在且形状正确**" ✓
      （审查 §3: 迁移可保留兼容层 ✓ 同一精神 ✓）
    """
    import io as _io
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    out = []
    for f in ("src/train/train.py", "src/train/trainer.py"):
        p = _os.path.join(root, f)
        if _os.path.exists(p):
            out.append(_io.open(p, encoding="utf-8").read())
    return "\n".join(out)



class _Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Log:
    def __init__(self):
        self.msgs = []

    def info(self, m, *a):
        self.msgs.append(m % a if a else m)

    def warning(self, m, *a):
        self.msgs.append("WARN: " + (m % a if a else m))


# ─────────────── ① 默认全关 ───────────────
def test_defaults_are_off():
    d = TEL.Diagnostics(_Args(), _Log())
    assert d.enabled is False
    assert d.describe() == "diagnostics=off"
    d.maybe_start(0)
    d.tick()                                    # 全关时必须完全无副作用 ✓
    assert d.prof.active is False and d.mem.on is False


def test_missing_args_are_safe():
    d = TEL.Diagnostics(object(), None)         # 无任何属性 ⇒ 走 getattr 缺省 ✓
    assert d.enabled is False and d.describe() == "diagnostics=off"


# ─────────────── ② 窗口语义 ───────────────
def test_profiler_window_skip_and_n(monkeypatch):
    started = []

    class _FakeProf:
        def __init__(self, **kw):
            started.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self):
            return []

    monkeypatch.setattr(TEL.torch.profiler, "profile", _FakeProf)
    lg = _Log()
    p = TEL.StepProfiler(_Args(profile_steps=3, profile_skip=5), lg)
    for i in range(5):
        p.maybe_start(i)
    assert started == [], "skip 之前不得启动 ✗"
    p.maybe_start(5)
    assert len(started) == 1 and p.active
    p.maybe_start(6)
    assert len(started) == 1, "已在窗口内不得重复启动 ✗"
    assert "[profile] 开始算子级 profile: 步 5 起 3 步" in lg.msgs[0]


def test_profiler_tick_prints_after_n_steps(monkeypatch):
    class _Avg(list):
        def table(self, sort_by=None, row_limit=None):
            return f"TABLE({sort_by},{row_limit})"

    class _FakeProf:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self):
            return _Avg()

    monkeypatch.setattr(TEL.torch.profiler, "profile", _FakeProf)
    lg = _Log()
    # ⚠ skip=0 会被 or 20 兜底成 20 ✗（原代码怪癖 ✓ 忠实保留 ✓）
    p = TEL.StepProfiler(_Args(profile_steps=2, profile_skip=1), lg)
    p.maybe_start(1)
    p.tick()
    assert not any("[profile] N=" in m for m in lg.msgs), "未到 N 步不得打印 ✗"
    p.tick()
    joined = "\n".join(lg.msgs)
    assert "[profile] N=2 步 | 排他 CUDA 忙 = " in joined, joined
    assert "逐算子 CUDA 时间 TOP20" in joined and "TABLE(self_cuda_time_total,20)" in joined
    assert "逐算子 CPU 时间 TOP15" in joined and "TABLE(self_cpu_time_total,15)" in joined
    assert p.active is False, "打印后必须关闭 ✓"


def test_memory_recorder_window(monkeypatch):
    calls = []

    class _FakeTCM:
        @staticmethod
        def _record_memory_history(**kw):
            calls.append(("record", kw))

        @staticmethod
        def _dump_snapshot(p):
            calls.append(("dump", p))

    # ⚠ import torch.cuda.memory as _tcm 走**包属性**解析
    #   ⇒ 补 sys.modules **无效** ✗ 必须补属性 ✓（实测确认 ✓）
    monkeypatch.setattr(TEL.torch.cuda, "memory", _FakeTCM)
    lg = _Log()
    m = TEL.MemoryHistoryRecorder(_Args(profile_memory_steps=2, profile_memory_skip=1), lg)
    m.maybe_start(0)
    assert not m.on
    m.maybe_start(1)
    assert m.on and calls[0][0] == "record" and calls[0][1] == {"max_entries": 300000}
    m.tick()
    assert not any(c[0] == "dump" for c in calls), "未到 N 步不得写快照 ✗"
    m.tick()
    assert any(c[0] == "dump" for c in calls), "到 N 步必须写快照 ✓"
    assert any(c[0] == "record" and c[1] == {"enabled": None} for c in calls), \
        "收尾必须关闭记录（enabled=None ✓）"
    assert m.on is False


def test_memory_snapshot_failure_is_nonfatal(monkeypatch):
    class _BadTCM:
        @staticmethod
        def _record_memory_history(**kw):
            pass

        @staticmethod
        def _dump_snapshot(p):
            raise RuntimeError("boom")

    monkeypatch.setattr(TEL.torch.cuda, "memory", _BadTCM)
    lg = _Log()
    m = TEL.MemoryHistoryRecorder(_Args(profile_memory_steps=1, profile_memory_skip=1), lg)
    m.maybe_start(1)
    m.tick()                                    # 不得抛出 ✓
    assert any("快照失败" in x for x in lg.msgs), lg.msgs


# ─────────────── ④ 顺序 ───────────────
def test_tick_order_profiler_then_memory():
    order = []

    class _P:
        def tick(self):
            order.append("prof")

    class _M:
        def tick(self):
            order.append("mem")

    d = TEL.Diagnostics(_Args(), None)
    d.prof, d.mem = _P(), _M()
    d.tick()
    assert order == ["prof", "mem"], "顺序必须与旧代码一致 ✗（先 profiler ✓ 再 分配历史 ✓）"


def test_maybe_start_order():
    order = []

    class _P:
        def maybe_start(self, i):
            order.append("prof")

    class _M:
        def maybe_start(self, i):
            order.append("mem")

    d = TEL.Diagnostics(_Args(), None)
    d.prof, d.mem = _P(), _M()
    d.maybe_start(0)
    assert order == ["prof", "mem"]


# ─────────────── ⑤ 纯观测边界 ───────────────
def test_module_is_pure_observation():
    """★ 边界 ✓: **诊断类**（StepProfiler / MemoryHistoryRecorder / Diagnostics）
    不得涉及训练语义 ✗。

    ⚠ 检查范围必须限定在**这三个类** ✗ —— 模块后来又收了
      verage_window（**纯计算** ✓），它合法地使用 "loss" 等**字典键** ✓
      （它只是指标名 ✓ 不是损失计算 ✗）。
      旧版本扫**全模块** ⇒ 新函数一加就误报 ✗（实测如此 ✓）。
    """
    src = io.open("src/train/telemetry.py", encoding="utf-8").read()
    tree = ast.parse(src)
    targets = {"StepProfiler", "MemoryHistoryRecorder", "Diagnostics"}
    code = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in targets:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = list(sub.body)
                    if body and isinstance(body[0], ast.Expr) and \
                            isinstance(body[0].value, ast.Constant) and \
                            isinstance(body[0].value.value, str):
                        body = body[1:] or [ast.Pass()]
                    code.append(ast.unparse(ast.Module(body=body, type_ignores=[])))
    joined = "\n".join(code)
    for bad in ("backward", "zero_grad", "opt.step", "clip_grad", "state_dict"):
        assert bad not in joined, f"诊断类不得涉及 {bad} ✗（纯观测 ✓）"
    assert code, "未找到诊断类 ✗（测试本身失效 ✗）"


def test_train_py_no_longer_owns_profiler_state():
    t = _read_train_src()
    assert "_diag = _Diagnostics(args, logger)" in t, "未构造 Diagnostics ✗"
    assert "_diag.maybe_start(batch_idx)" in t, "步首未接线 ✗"
    assert "_diag.tick()" in t, "窗口终点未接线 ✗"
    for bad in ("_PROF_OBJ", "_PROF_CNT", "_PROF_N", "_MEM_ON", "_MEM_CNT", "_MEM_N"):
        assert bad not in t, f"train.py 仍持有诊断状态 `{bad}` ✗（应已迁入 telemetry ✓）"
    assert "torch.profiler.profile(" not in t, "train.py 不应再直接建 profiler ✗"


def test_describe_reports_window():
    d = TEL.Diagnostics(_Args(profile_steps=5, profile_skip=3,
                              profile_memory_steps=4, profile_memory_skip=2), None)
    s = d.describe()
    assert d.enabled is True
    assert "profile_steps=5 skip=3" in s and "profile_memory_steps=4 skip=2" in s, s


# ─────────────── ⑥ 记录一处**原有怪癖**（忠实保留 ✓ 不擅自改 ✗）───────────────
def test_zero_skip_is_coerced_to_default_by_or_quirk():
    """★ 原有怪癖 ✓: int(getattr(args, k, D) or D) ⇒ **传 0 会被兜底成 D** ✗

    --profile-skip 0 / --profile-memory-skip 0 因此**无法表达"不跳过"** ✗
      （实际会被当成 20 / 10 ✓）
    本模块**逐字保留**该行为 ✓（重构不得顺手改变语义 ✗）⇒ 用测试锁定 ✓
    若将来要修 ⇒ 那是**行为变更** ✗ 应单独决策 ✓（且要改 CLI 文档 ✓）
    """
    p = TEL.StepProfiler(_Args(profile_steps=5, profile_skip=0), None)
    assert p.skip == 20, "传 0 应被 or 兜底成 20 ✓（原行为 ✓）"
    m = TEL.MemoryHistoryRecorder(_Args(profile_memory_steps=5, profile_memory_skip=0), None)
    assert m.skip == 10, "传 0 应被 or 兜底成 10 ✓（原行为 ✓）"
    # 非零值正常生效 ✓
    assert TEL.StepProfiler(_Args(profile_steps=5, profile_skip=3), None).skip == 3
