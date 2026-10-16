# -*- coding: utf-8 -*-
"""`training_loop` 的**局部变量初始化**守卫（新版审查 §二 ✓ 2026-10-16 ✓）

★★ 审查原文（批评完全成立 ✓ 必须正面回应 ✓）:
  "`src/train/trainer.py:963–983` … 有两个控制变量没有初始化:
     `early_stop_stopped`（仅第 963 行）· `_train_fatal`（仅异常处理第 972 行）
   … 原 `train.py` 中对应变量是初始化过的，只是迁移时没有作为局部状态处理。" ✓
  "这个问题足以说明：迁移说明中的'AST 验证循环状态只读，因此逐字搬移等价'**并不成立**。
   AST 比对**没有覆盖函数内部变量的控制流初始化和异常路径**。" ✓
  "建议…增加三个测试：正常完成、early stop、训练中抛异常。
   **仅靠代码能够导入、通过语法检查或完成几个训练 step，不足以验证这个问题**。" ✓

★★★ 为什么真机测试**掩盖**了它（必须记录 ✓）:
  我的所有真机测试都传了 `--max-steps` ✓ ⇒ `if max_steps …: break` 先执行 ✓
  ⇒ **`if early_stop_stopped:` 的读取点被跳过** ✓ ⇒ bug 不可见 ✗✗
  ⇒ ★ **真实长训练（不设 `--max-steps`）会在第一个 epoch 边界崩** ✗✗

★ 本测试的做法（良好工程 ✓ 便宜且决定性 ✓）:
  用 AST 检查 `training_loop` 里这两个名字**是否在函数体顶层被赋值** ✓
  （= 任何控制流之前 ✓）—— 这正是不依赖"跑几步"的**静态**验证 ✓
  ★ 同时**验证修复有效** ✓: 若有人再把初始化删掉 ✓ 立刻 FAIL ✓
"""
import ast
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "src", "train", "trainer.py")
GUARDED = ("early_stop_stopped", "_train_fatal")


def _fn():
    tree = ast.parse(io.open(TRAINER, encoding="utf-8").read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "training_loop")


def _top_level_assigned(fn):
    """★ **只**取 `fn.body` **直接**赋值语句的目标名 ✓（不钻进嵌套块 ✗）。

    ⚠ 为什么不能用 `ast.walk(st)`（本测试首版就错在这里 ✗ 被元测试抓到 ✓）:
      `ast.walk` 会**递归进入**嵌套块 ✗ ⇒ `except:` 里的 `_train_fatal = e`
      会被误判成"函数体顶层赋值" ✗✗ ⇒ 守卫形同虚设 ✗
      ⇒ 必须只看 `fn.body` 的**直接**元素 ✓
    """
    out = {}
    for st in fn.body:
        targets = []
        if isinstance(st, ast.Assign):
            targets = st.targets
        elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
            targets = [st.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.setdefault(n.id, st.lineno)
    return out


def test_guarded_vars_are_assigned_at_function_top_level():
    """★ 两个变量必须在**函数体顶层**（任何控制流之前）被赋值 ✓。

    ★ 为什么要求"顶层"而不是"存在某处赋值" ✗:
      `_train_fatal = e` 在 `except` 里 ✓ 也算"有赋值" ✓ 但**异常不发生**就到不了 ✗
      `early_stop_stopped = True` 在 `while` 体的 `if` 里 ✓ 同理 ✗
      ⇒ 只有**函数体顶层**的赋值才能保证"任何路径都先赋值" ✓✓
    """
    fn = _fn()
    top = _top_level_assigned(fn)
    missing = [v for v in GUARDED if v not in top]
    assert not missing, (
        "`training_loop` 缺少**函数体顶层**初始化 ✗: %s\n"
        "（新版审查 §二 的 P0 ✓ 真实长训练会在第一个 epoch 边界 "
        "`UnboundLocalError` 崩溃 ✗✗）\n"
        "修法: 在 `training_loop` 开头加 `early_stop_stopped = False` / "
        "`_train_fatal = None` ✓" % missing)


def test_assignment_precedes_every_read():
    """★ 顶层的赋值必须出现在**所有读取之前** ✓（源序 ✓ 同一函数作用域 ✓）。"""
    fn = _fn()
    top = _top_level_assigned(fn)
    for var in GUARDED:
        assert var in top, "`%s` 未在顶层赋值 ✗" % var
        reads = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Load)]
        assert reads, "`%s` 从未被读取 ✗（测试失效 ✓ 请核对变量名）" % var
        assert top[var] < min(reads), (
            "`%s` 的顶层赋值 (L%d) 晚于最早读取 (L%d) ✗" %
            (var, top[var], min(reads)))


def test_the_two_original_bugs_would_be_caught():
    """★★ **元测试** ✓: 证明本守卫**真能**抓到审查 §二 报告的那两个写法 ✓。

    做法: 构造一段**等价于旧代码**的源码 ✓ 断言守卫判其"不合格" ✓
      · `_train_fatal` 只在 `except` 里赋值 ✗
      · `early_stop_stopped` 只在 `while` 体内的 `if` 里赋值 ✗
    ⇒ ★ 若本元测试失败 ✓ 说明守卫形同虚设 ✗（等于没有回归保护 ✗）
    """
    BAD = (
        "def f():\n"
        "    while True:\n"
        "        try:\n"
        "            if x:\n"
        "                early_stop_stopped = True\n"
        "                break\n"
        "        except Exception as e:\n"
        "            _train_fatal = e\n"
        "            break\n"
        "        if early_stop_stopped:\n"
        "            break\n"
        "    return _train_fatal\n"
    )
    fn = next(n for n in ast.parse(BAD).body if isinstance(n, ast.FunctionDef))
    top = _top_level_assigned(fn)
    assert "early_stop_stopped" not in top and "_train_fatal" not in top, (
        "元测试失效 ✗: 旧代码写法竟被判为合格 ✗ ⇒ 守卫抓不到 §二 的 bug ✗\n"
        "（★ 这正说明「只判存在某处赋值」是不够的 ✓ 必须判**函数体顶层** ✓）")
