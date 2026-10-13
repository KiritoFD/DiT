# -*- coding: utf-8 -*-
"""tests/test_eval_protocol.py — 评测身份（eval_id）的不变量测试
（2026-10-13 [审查 §4.3]："同一 eval_id 可以安全复用；任意字段不同，都必须重新生成"）

这些测试保护的正是本项目反复踩过的坑:
  · 同 step 不同模型/CFG/TTF ⇒ 复用旧图 ✗
  · 残留图/错图被当成"已完成" ✗
  · 改了协议（或产物）却仍被判为可复用 ✗

运行: python -m pytest tests/test_eval_protocol.py -q
      或  python tests/test_eval_protocol.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.protocol import (EvalRequest, eval_id, is_complete,        # noqa: E402
                               sha256_tree, write_done, write_protocol)

H40 = "a" * 64          # 合法的 64 位十六进制（sha256 形状 ✓）
H40B = "b" * 64


def base_req(**kw) -> EvalRequest:
    d = dict(checkpoint_sha256=H40, vae_sha256=H40B, vae_scale=0.18215, vae_shift=0.0,
             sampler="heun", num_steps=50, cfg=1.0, gamma=0.0, seed=0, ema=True,
             code_revision="deadbee", dataset_sha256="c" * 64,
             conditions_sha256="d" * 64, sets_spec="eval200fix:187")
    d.update(kw)
    return EvalRequest(**d)


# ─────────────────── ① 同一协议 ⇒ 同一 id（幂等 ✓）───────────────────
def test_same_request_same_id():
    a, b = base_req(), base_req()
    assert a.eval_id() == b.eval_id()
    assert len(a.eval_id()) == 64


# ─────────────────── ② ★ 任一字段不同 ⇒ id 必不同（审查原话 ✓✓）───────────────────
def test_any_field_change_changes_id():
    """逐字段验证 —— **一个字段都不能漏** ✓（含浮点/布尔/字符串 ✓）"""
    base = base_req().eval_id()
    changes = {
        "checkpoint_sha256": "e" * 64,
        "vae_sha256": "f" * 64,
        "vae_scale": 0.18315,
        "vae_shift": -0.03,
        "sampler": "euler",
        "num_steps": 100,
        "cfg": 4.0,
        "gamma": 0.5,
        "seed": 1,
        "ema": False,
        "code_revision": "beefbee",
        "dataset_sha256": "1" * 64,
        "conditions_sha256": "2" * 64,
        "sets_spec": "eval200fix:187,seen:20",
    }
    for k, v in changes.items():
        r = base_req(**{k: v})
        assert r.eval_id() != base, f"字段 {k} 改变后 eval_id **必须**不同 ✗（审查要求 ✓）"
    # extra 也必须进入身份 ✓
    assert base_req(extra={"font": "simkai"}).eval_id() != base


# ─────────────────── ③ 空/缺身份不许冒充（缺失就停止 ✓）───────────────────
def test_missing_identity_is_rejected():
    for k in ("checkpoint_sha256", "vae_sha256"):
        try:
            eval_id(base_req(**{k: ""}))
            raise AssertionError(f"{k} 为空时必须报错 ✓")
        except ValueError:
            pass
    try:
        eval_id(base_req(schema_version=999))
        raise AssertionError("schema_version 不一致必须报错 ✓")
    except ValueError:
        pass


# ─────────────────── ④ 完成判定: 不只看 DONE 存在 ✓✓ ───────────────────
def test_completion_requires_matching_identity_and_intact_artifacts():
    req = base_req()
    d = os.path.join(tempfile.mkdtemp(), "evaluations", req.eval_id())
    # (a) 什么都没写 ⇒ 未完成 ✓
    ok, why = is_complete(req, d)
    assert not ok and "DONE" in why, why
    # (b) 写协议 + 产物 + DONE ⇒ 完成 ✓
    write_protocol(req, d)
    import io
    io.open(os.path.join(d, "metrics.json"), "w", encoding="utf-8").write('{"ssim": 0.6}')
    write_done(req, d, n_samples=187)
    ok, why = is_complete(req, d)
    assert ok, why
    # (c) ★ **同 step 但协议不同**（换 CFG）⇒ 绝不能复用 ✓✓
    ok2, why2 = is_complete(base_req(cfg=4.0), d)
    assert not ok2 and "eval_id" in why2, why2
    # (d) 产物被改 ⇒ 失效 ✓
    io.open(os.path.join(d, "metrics.json"), "w", encoding="utf-8").write('{"ssim": 0.9}')
    ok3, why3 = is_complete(req, d)
    assert not ok3 and "哈希不符" in why3, why3


# ─────────────────── ⑤ 残留图不被误判（产物缺失 ⇒ 失效 ✓）───────────────────
def test_missing_artifact_invalidates_completion():
    req = base_req()
    d = os.path.join(tempfile.mkdtemp(), "evaluations", req.eval_id())
    write_protocol(req, d)
    import io
    io.open(os.path.join(d, "metrics.json"), "w", encoding="utf-8").write("{}")
    write_done(req, d, n_samples=3)
    assert is_complete(req, d)[0]
    os.remove(os.path.join(d, "metrics.json"))          # 模拟"图/表被清掉但 DONE 还在" ✗
    ok, why = is_complete(req, d)
    assert not ok and "缺失" in why, why


# ─────────────────── ⑥ 目录指纹: 内容变 ⇒ 指纹变（数据集身份 ✓）───────────────────
def test_tree_hash_is_content_addressed():
    r = tempfile.mkdtemp()
    try:
        import io
        io.open(os.path.join(r, "a.csv"), "w", encoding="utf-8").write("x,y\n1,2\n")
        h1 = sha256_tree(r)
        io.open(os.path.join(r, "b.csv"), "w", encoding="utf-8").write("x,y\n3,4\n")
        h2 = sha256_tree(r)
        assert h1 != h2, "新增文件后目录指纹必须变 ✓"
        # 同名文件改内容 ⇒ 也变 ✓
        before = sha256_tree(r)
        io.open(os.path.join(r, "a.csv"), "w", encoding="utf-8").write("x,y\n9,9\n")
        assert sha256_tree(r) != before, "内容变化必须改变指纹 ✓"
        # 幂等 ✓
        assert sha256_tree(r) == sha256_tree(r)
        # 不存在的目录必须报错 ✓（不许静默返回空指纹 ✗）
        try:
            sha256_tree(os.path.join(r, "nope"))
            raise AssertionError("不存在目录必须报错 ✓")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(r, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("  ✓ %s" % fn.__name__)
        except Exception as e:                                        # noqa: BLE001
            bad += 1
            print("  ✗ %s -> %s: %s" % (fn.__name__, type(e).__name__, str(e)[:110]))
    print("\n  %d/%d 通过" % (len(fns) - bad, len(fns)))
    sys.exit(1 if bad else 0)
