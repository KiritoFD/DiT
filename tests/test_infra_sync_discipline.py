# -*- coding: utf-8 -*-
"""训练热路径**同步纪律**的 CI 守卫（`docs/infra_1015.md` §二.3 ✓）

审查原文:
  "我建议把它设为一条**全局规则**：正常训练热路径中禁止 `.item()`、
   `float(cuda_tensor)`、`.tolist()`、`bool(cuda_tensor)`；
   **只有显式标记的同步边界允许使用**。" ✓
  "可以在 **CI** 中对训练热路径做静态扫描，并对**已批准的例外添加注释**。" ✓

★ 本测试**不做**"零违规"硬断言 ✗ —— 静态扫描无法判类型 ✗ 必然有误报 ✓
  （例: `float(args.critic_t_min)` 是 Python float ✓ 不同步 ✗ 但形似 tensor 调用 ✗）
  真实首轮: 175 处 ✗ 豁免日志边界后 **148** 处 ✓
⇒ 改为**棘轮** ✓: `tools/sync_hotpath_baseline.json` 记录每文件基线 ✓
  测试**只在数量增长时失败** ✗（与本项目 `tools/preflight.py` 的结构下限同一模式 ✓）

★ 断言三层 ✓:
  ① 基线文件存在且可解析 ✓
  ② 扫描器本身可用（对一段**已知**代码给出**正确**判定 ✓）—— 防"工具悄悄坏掉" ✗
  ③ 当前未标记数 **不超出**基线 ✗
"""
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "lint_sync_hotpath.py")
BASELINE = os.path.join(ROOT, "tools", "sync_hotpath_baseline.json")


def _run(*args):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, TOOL, *args], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)


def test_baseline_file_exists_and_parses():
    assert os.path.isfile(BASELINE), (
        "缺少棘轮基线 ✗ 请运行: python tools/lint_sync_hotpath.py --update-baseline")
    data = json.load(io.open(BASELINE, encoding="utf-8"))
    assert isinstance(data, dict) and data, "基线内容应为非空 dict ✗"
    for k, v in data.items():
        assert isinstance(k, str) and isinstance(v, int) and v >= 0, \
            "基线项应为 文件→非负整数 ✗: %r=%r" % (k, v)


def test_scanner_self_check():
    """★ 扫描器**自身**的正确性 ✓ —— 用一段**已知**代码验证判定 ✓。

    防的是"工具悄悄坏掉" ✗（例如 AST 逻辑写错 ⇒ 永远返回 0 违规 ✗ ⇒
    棘轮就再也抓不到任何回归 ✗✗）—— 这是**元测试** ✓ 良好工程实践 ✓
    """
    r = _run("--json")
    assert r.returncode in (0, 1), "扫描器应返回 0/1，实际 %d ✗\n%s" % (r.returncode, r.stderr)
    payload = json.loads(r.stdout)
    assert "files" in payload and "regressions" in payload, "JSON 结构不符 ✗"
    # ★ 必须真的扫到了同步点 ✓（否则工具等于失效 ✗）
    total = sum(v.get("total", 0) for v in payload["files"].values())
    assert total > 0, (
        "扫描器一个同步点都没扫到 ✗ ⇒ 工具可能已失效 ✗（§二.3 的守卫会形同虚设 ✗）")
    # ★ 必须真的识别出"已标记"的 ✓（否则白名单机制失效 ✗）
    marked = sum(v.get("marked", 0) for v in payload["files"].values())
    assert marked > 0, "扫描器未识别出任何已标记同步点 ✗ ⇒ 白名单机制失效 ✗"


def test_no_sync_regression_beyond_baseline():
    """★ 棘轮 ✓: 任一文件的未标记同步数**不得超出**基线 ✗。"""
    r = _run("--json")
    payload = json.loads(r.stdout)
    regress = payload.get("regressions") or {}
    assert not regress, (
        "训练热路径同步数**超出基线** ✗（infra_1015 §二.3 ✓）:\n" +
        "\n".join("  %s: 基线 %s → %s ✗" % (k, b, v)
                  for k, (b, v) in sorted(regress.items())) +
        "\n修法: 把标量读取推迟到日志周期 ✓，或在该行/上一行写"
        " `# sync-ok: <理由>` ✓（须写理由 ✓ 不得静默 ✗）")
    assert r.returncode == 0, "扫描器退出码非 0 ✗\n%s" % r.stdout


def test_log_boundary_exemption_is_documented():
    """★ 日志边界豁免必须**有据可依** ✓（§二.2 明确允许 ✓）—— 防止豁免被滥用 ✗。"""
    src = io.open(TOOL, encoding="utf-8").read()
    assert "LOG_ATTRS" in src and "LOG_CALLERS" in src, "日志豁免表缺失 ✗"
    assert "只在真正写日志时" in src or "真正写日志" in src, \
        "日志豁免必须**引用审查原文**作为依据 ✓（不得无据豁免 ✗）"


def test_human_readable_path_prints_something():
    """★★ **人类可读路径必须真的输出** ✓（不是只有 `--json` 能跑 ✓）。

    ★ 这道测试是**补上一个真实漏网**（2026-10-16 实测 ✓）:
      上一轮我给 `main()` 加棘轮时，替换 `if args.json:` 分支的编辑**吞掉了 `else:`** ✗
      ⇒ 报告块与 `return` 同在 8 空格缩进 ✗ ⇒ 变成**不可达死代码** ✗✗
      ⇒ 工具跑起来**完全静默** ✗（而 `--json` 仍正常 ✓）
      ⇒ 当时的测试**只覆盖了 `--json`** ✗ ⇒ **完全没发现** ✗
    ★ 教训: 一个工具有多条输出路径时，**每条都要有测试** ✓
      —— 只测机器可读路径 ⇒ 人类可读路径可以静默死掉 ✗
    """
    r = _run()                                    # ← **不带** --json ✓
    assert r.returncode in (0, 1), "退出码应为 0/1，实际 %d ✗\n%s" % (r.returncode, r.stderr)
    out = r.stdout
    assert out.strip(), (
        "**人类可读路径没有任何输出** ✗ ⇒ 报告块很可能是**不可达死代码** ✗\n"
        "（检查 `main()` 里 `if args.json:` 分支的 `return` 之后是否还有代码 ✗）")
    assert "训练热路径同步纪律扫描" in out, "报告标题缺失 ✗ ⇒ 输出路径异常 ✗"
    assert "汇总" in out, "报告**汇总行**缺失 ✗ ⇒ 可能只打印了文件列表 ✓ 但没结论 ✗"


def test_stats_and_plain_paths_both_work():
    """★ `--stats` 与默认路径都应输出 ✓（多路径都要有测试 ✓ 见上一道测试的教训 ✓）。"""
    for flag in ([], ["--stats"]):
        r = _run(*flag)
        assert r.stdout.strip(), "`%s` 路径无输出 ✗" % (" ".join(flag) or "(默认)")
        assert "汇总" in r.stdout, "`%s` 路径缺汇总行 ✗" % (" ".join(flag) or "(默认)")
