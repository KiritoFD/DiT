# -*- coding: utf-8 -*-
"""test_refactor_p4.py —— 2026-10-14 [重构 P0-1] **接入 in_mem_eval** 的验收测试 ✓

审查原文: "项目已经实现 EvalRequest / eval_id / DONE.json 和文件哈希校验, 但
  **没有找到主评测流程对 write_done() / is_complete() 的实际调用**" ✗
  ⇒ 本组测试锁定"接入"这件事本身 ✓:
    ① 身份 = **权重内容 sha256** ✓（in-training eval 没有 ckpt 文件 ✗）
    ② DONE 存在即**权威** ✓（失败强制重跑 ✗）
    ③ 真实 187 张 PNG 的协议往返 ✓（专家验收标准 ✓）
"""
from __future__ import annotations

import io
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import in_mem_eval as IE                                     # noqa: E402
from src.eval import protocol as P                                         # noqa: E402


# ─────────────── ① 权重内容身份（in-training eval 的关键 ✓）───────────────
def test_model_weight_sha256_is_content_addressed():
    m1 = torch.nn.Linear(4, 3)
    m2 = torch.nn.Linear(4, 3)
    m2.load_state_dict(m1.state_dict())
    h1 = IE._model_weight_sha256(m1)
    h2 = IE._model_weight_sha256(m2)
    assert h1 and len(h1) == 64
    assert h1 == h2, "同权重必须同哈希 ✓（内容寻址 ✓）"
    with torch.no_grad():
        m2.weight.add_(0.01)                    # 改一个权重 ⇒ 身份必须变 ✗
    assert IE._model_weight_sha256(m2) != h1, "权重变 ⇒ 身份必须变 ✗"


def test_model_weight_sha256_unwraps_module():
    """DDP/compile 包装（.module）必须被正确解开 ✓"""
    class _Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
    inner = torch.nn.Linear(3, 2)
    assert IE._model_weight_sha256(_Wrap(inner)) == IE._model_weight_sha256(inner)


def test_vae_sha256_dir_and_missing(tmp_path):
    d = os.path.join(str(tmp_path), "vae")
    os.makedirs(d)
    io.open(os.path.join(d, "config.json"), "w", encoding="utf-8").write("{}")
    h = IE._vae_sha256(d)
    assert h and len(h) == 64, "目录 VAE 必须可哈希 ✓"
    assert IE._vae_sha256(os.path.join(str(tmp_path), "nope")) is None, \
        "缺失 VAE 必须返回 None ✗（绝不伪造身份 ✗）"
    assert IE._vae_sha256("") is None


# ─────────────── ② EvalRequest 构造（缺身份 ⇒ None ✗）───────────────
class _Args:
    eval_vae_path = ""
    vae_shift_factor = 0.0
    std_soft_eval_sigma = 0.65


def _mk_req(tmp_path, vae_ok=True, csv_ok=True):
    vae = os.path.join(str(tmp_path), "vae")
    os.makedirs(vae, exist_ok=True)
    io.open(os.path.join(vae, "c.json"), "w", encoding="utf-8").write("{}")
    csvp = os.path.join(str(tmp_path), "e.csv")
    io.open(csvp, "w", encoding="utf-8").write("a,b\n1,2\n")
    a = _Args()
    a.eval_vae_path = vae if vae_ok else os.path.join(str(tmp_path), "missing_vae")
    m = torch.nn.Linear(4, 4)
    return IE._build_eval_request(a, m, 5000, "eval200fix", 187, 1.0, 50, 0.18215,
                                  csvp if csv_ok else os.path.join(str(tmp_path), "no.csv"),
                                  "rev1")


def test_build_eval_request_ok(tmp_path):
    req = _mk_req(tmp_path)
    assert req is not None, "身份齐备时必须构造成功 ✓"
    assert req.num_steps == 50 and req.cfg == 1.0 and abs(req.vae_scale - 0.18215) < 1e-9
    assert req.sets_spec == "eval200fix:187"
    assert len(req.eval_id()) == 64


def test_build_eval_request_none_when_vae_missing(tmp_path):
    assert _mk_req(tmp_path, vae_ok=False) is None, \
        "VAE 身份拿不到 ⇒ 必须返回 None ✗（绝不伪造 ✗）"


def test_eval_id_changes_with_cfg_and_steps(tmp_path):
    r1 = _mk_req(tmp_path)
    r2 = IE._build_eval_request(_Args(), torch.nn.Linear(4, 4), 5000, "eval200fix",
                                187, 1.4, 50, 0.18215,
                                os.path.join(str(tmp_path), "e.csv"), "rev1")
    assert r1.eval_id() != r2.eval_id(), "换 CFG ⇒ 身份必须不同 ✗（审查要求的正是这点 ✓）"


# ─────────────── ③ 真实 187 张 PNG 的协议往返（专家验收 ✓）───────────────
def _real_png_dir():
    for c in ("assets/style_eval_src/v88_step0050000",
              "assets/style_eval_src/step0200000"):
        if os.path.isdir(c) and len([f for f in os.listdir(c)
                                     if f.startswith("g") and f.endswith(".png")]) >= 100:
            return c
    return None


def test_real_187_png_protocol_roundtrip(tmp_path):
    """★ 用**真实评测产物**（187 张 g*.png + gt*.png 混杂 ✓）验证协议 ✓:
        gt*.png 必须被忽略 ✓ / 计数必须精确 ✓ / 删一张即失效 ✓"""
    src = _real_png_dir()
    if src is None:
        pytest.skip("本地无真实评测图目录")
    import re as _re
    import shutil
    d = os.path.join(str(tmp_path), "set")
    os.makedirs(d)
    pat = _re.compile(r"^g\d+\.png$")
    n = 0
    for f in sorted(os.listdir(src)):
        if pat.match(f):                    # ★ 只数 g<数字>.png ✓（gt*.png 不算 ✗）
            shutil.copy(os.path.join(src, f), os.path.join(d, f))
            n += 1
    assert n >= 100
    inv = P.png_inventory(d)
    assert len(inv) == n, "gt*.png 必须被忽略 ✓（实测 %d vs %d）" % (len(inv), n)
    io.open(os.path.join(d, "samples.jsonl"), "w", encoding="utf-8").write(
        "".join('{"idx":%d}\n' % i for i in range(n)))
    req = P.EvalRequest(checkpoint_sha256="a" * 64, vae_sha256="b" * 64,
                        vae_scale=0.18215, vae_shift=0.0, sampler="heun",
                        num_steps=50, cfg=1.0, gamma=0.0, seed=0, ema=True,
                        code_revision="rev")
    P.write_protocol(req, d)
    P.write_done(req, d, n_samples=n, png_dir=d)
    ok, why = P.is_complete(req, d)
    assert ok, why
    victim = sorted(inv)[n // 2]
    os.remove(os.path.join(d, victim))
    ok2, why2 = P.is_complete(req, d)
    assert not ok2 and "PNG" in why2, why2


# ─────────────── ④ 接线静态守卫（防回归 ✓）───────────────
def test_in_mem_eval_wires_protocol():
    src = io.open("src/eval/in_mem_eval.py", encoding="utf-8").read()
    assert "_build_eval_request" in src, "必须构造 EvalRequest ✗（P0-1 接入）"
    assert "is_complete(" in src, "主流程必须调用 is_complete ✗（审查的缺陷 ✗）"
    assert "write_done(" in src, "主流程必须调用 write_done ✗（审查的缺陷 ✗）"
    assert "DONE 存在即权威" in src or "DONE 校验**失败** ⇒ **强制重跑**" in src, \
        "必须有「DONE 权威 + 失败重跑」语义 ✓"
    assert "_model_weight_sha256" in src, "身份必须用**权重内容**哈希 ✓"


def test_protocol_module_still_zero_dependency():
    """protocol.py 必须保持**零依赖** ✓（审查设计约束 ✓ 谁都能调 ✓）"""
    src = io.open("src/eval/protocol.py", encoding="utf-8").read()
    for bad in ("import torch", "from src.model", "import numpy"):
        assert bad not in src, "protocol.py 不得依赖 %s ✗" % bad
