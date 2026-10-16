# -*- coding: utf-8 -*-
"""test_refactor_p0.py —— 2026-10-14 重构 P0/P1 修复的**验收测试** ✓

覆盖审查报告里**已复现**的缺陷与验收标准:
  P0-1 评测身份: 缺图/删图/换图/改协议/伪造空 DONE ⇒ **必须判未完成** ✗
       （审查实测: 旧实现对这些全部返回 True ✗✗）
  P0-2 数据加载: 未初始化内存不得进 batch（代码层: np.empty 已清零 ✓ 见静态检查）
  P0-3 Critic 查表: 缺失 (书体,字符) **必须报错**, 不许 clip 到第 0 行 ✗
  P1-6 Critic 同步: forward 返回 **detached tensor** 而非 float ✓
  P1-7 稳定键: set_char_lookup 用 Unicode codepoint 而非 hash() ✓
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import protocol as P                                    # noqa: E402

REQ_KW = dict(
    checkpoint_sha256="a" * 64,
    vae_sha256="b" * 64,
    vae_scale=0.18215,
    vae_shift=0.0,
    sampler="heun",
    num_steps=50,
    cfg=1.0,
    gamma=0.0,
    seed=0,
    ema=True,
    code_revision="testrev",
)




def _mk_metrics(d, n=3):
    """★ 新契约 ✓: 合法完成的 DONE 必须登记本集 `metrics.json` ✓（2026-10-16 ✓）。"""
    import json as _json
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as _f:
        _json.dump({"n": n, "ssim": 0.5, "mse": 0.5}, _f)


def _req(**kw):
    d = dict(REQ_KW)
    d.update(kw)
    return P.EvalRequest(**d)


def _mk_pngs(d, n, fill=b"\x89PNG"):
    for i in range(n):
        with open(os.path.join(d, "g%d.png" % i), "wb") as f:
            f.write(fill + bytes([i % 256]))


def _mk_manifest(d, name="samples.jsonl", n=3):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i}) + "\n")


# ─────────────────────────── P0-1 评测身份 ───────────────────────────
def test_write_done_rejects_missing_manifest(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_pngs(d, 3)
    with pytest.raises(RuntimeError, match="manifest"):
        _mk_metrics(d)
        P.write_done(req, d, n_samples=3)          # 无 samples.csv/jsonl ⇒ 必须报错 ✗


def test_write_done_rejects_wrong_png_count(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=5)
    _mk_pngs(d, 3)                                  # 声称 5 张, 实际 3 张 ⇒ 报错 ✗
    with pytest.raises(RuntimeError, match="PNG"):
        _mk_metrics(d)
        P.write_done(req, d, n_samples=5)


def test_write_done_ok_then_is_complete(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=3)
    _mk_pngs(d, 3)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=3)
    ok, why = P.is_complete(req, d)
    assert ok, why


def test_is_complete_rejects_forged_empty_done(tmp_path):
    """★ 审查的**最小复现** ✗: 有 protocol + 声称 187 样本的 DONE, 但一张 PNG 都没有 ⇒
    旧实现返回 True ✗✗；修复后**必须 False** ✓"""
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    forged = {
        "eval_id": req.eval_id(),
        "protocol_sha256": P.sha256_file(P.protocol_path(d)),
        "schema_version": P.SCHEMA_VERSION,
        "n_samples": 187,
        "artifacts": {},
    }
    with open(P.done_path(d), "w", encoding="utf-8") as f:
        json.dump(forged, f)
    ok, why = P.is_complete(req, d)
    assert not ok, "伪造空 DONE 被判完成 ✗✗（P0-1 未修复）"
    assert ("manifest" in why) or ("PNG" in why), why


def test_is_complete_detects_deleted_png(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=4)
    _mk_pngs(d, 4)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=4)
    assert P.is_complete(req, d)[0]
    os.remove(os.path.join(d, "g2.png"))            # 删一张 ⇒ 必须失效 ✗
    ok, why = P.is_complete(req, d)
    assert not ok and "PNG" in why, why


def test_is_complete_detects_modified_png(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=4)
    _mk_pngs(d, 4)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=4)
    with open(os.path.join(d, "g1.png"), "ab") as f:
        f.write(b"tampered")                        # 换内容 ⇒ 必须失效 ✗
    ok, why = P.is_complete(req, d)
    assert not ok and "哈希" in why, why


def test_is_complete_detects_extra_png(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=4)
    _mk_pngs(d, 4)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=4)
    _mk_pngs(d, 5)                                  # 多一张 g4.png ⇒ 集合不符 ✗
    ok, why = P.is_complete(req, d)
    assert not ok and "PNG" in why, why


def test_is_complete_detects_protocol_change(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=3)
    _mk_pngs(d, 3)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=3)
    P.write_protocol(_req(cfg=1.4), d)              # 采样参数变了 ⇒ 协议哈希变 ⇒ 失效 ✗
    ok, why = P.is_complete(req, d)
    assert not ok, why


def test_is_complete_detects_eval_id_mismatch(tmp_path):
    d = str(tmp_path)
    req = _req()
    P.write_protocol(req, d)
    _mk_manifest(d, n=3)
    _mk_pngs(d, 3)
    _mk_metrics(d)
    P.write_done(req, d, n_samples=3)
    other = _req(checkpoint_sha256="c" * 64)        # 别的 checkpoint ⇒ 不可复用 ✗
    ok, why = P.is_complete(other, d)
    assert not ok and "eval_id" in why, why


def test_png_inventory_only_counts_gN(tmp_path):
    d = str(tmp_path)
    _mk_pngs(d, 2)
    with open(os.path.join(d, "gt0.png"), "wb") as f:
        f.write(b"x")                               # gt*.png 不算 ✓
    with open(os.path.join(d, "g.png"), "wb") as f:
        f.write(b"x")                               # 无数字不算 ✓
    assert set(P.png_inventory(d)) == {"g0.png", "g1.png"}


# ─────────────────────────── P0-3 / P1-6 / P1-7 ───────────────────────────
def _fake_critic(rows_keys, bank_n=5):
    """构造一个**不加载 ckpt** 的 CriticLoss 实例 ✓（只测查表逻辑 ✓）。"""
    import torch

    from src.loss.critic_loss import CriticLoss
    cl = CriticLoss.__new__(CriticLoss)
    torch.nn.Module.__init__(cl)                    # ★ 必须初始化 nn.Module ✓
    #   （否则 `cl.model = ...` 抛 AttributeError: cannot assign module before init ✗）
    cl.g_std_bank = torch.arange(bank_n * 2 * 2, dtype=torch.float32).reshape(
        bank_n, 2, 2)                               # 每行 [2,2]（真实 bank 为 [M,64,64] ✓）
    cl._key2row = {k: i for i, k in enumerate(rows_keys)}
    cl.s2i = {"楷": 0, "行": 1}
    cl.slots = ["A·楷", "B·行"]
    cl._lut = {}
    return cl


def test_critic_g_for_raises_on_missing_key():
    cl = _fake_critic(["楷·永", "行·山"])
    with pytest.raises(KeyError, match="strict_lookup"):
        cl._g_for(["楷", "楷"], ["永", "缺"], "cpu")     # 「缺」不在 bank ⇒ 报错 ✗


def test_critic_g_for_rows_raises_on_out_of_range():
    cl = _fake_critic(["楷·永"])
    with pytest.raises(IndexError, match="strict_lookup"):
        cl._g_for_rows([-1, 0], "cpu")                  # -1 不得 clip 到第 0 行 ✗
    with pytest.raises(IndexError, match="strict_lookup"):
        cl._g_for_rows([99], "cpu")                     # 越界 ⇒ 报错 ✗


def test_critic_g_for_ok_path():
    cl = _fake_critic(["楷·永", "行·山"])
    g = cl._g_for(["楷", "行"], ["永", "山"], "cpu")
    assert tuple(g.shape) == (2, 1, 2, 2)


def test_critic_coverage_audit_raises():
    cl = _fake_critic(["楷·永"])
    with pytest.raises(KeyError, match="覆盖率审计"):
        cl.coverage_audit(["楷", "行"], ["永", "山"])


def test_critic_forward_returns_detached_tensors():
    """P1-6 ✓: parts 必须是 **tensor**（不是 float ✗）⇒ 无每步 CUDA 同步 ✓"""
    import torch
    cl = _fake_critic(["楷·永"])
    cl.model = None                                 # 用不到（直接看 forward_g 的返回构造）
    # 手工走 forward_g 的返回语句路径: 用一个极小的假 model ✓
    class _M(torch.nn.Module):
        def forward(self, z, g, si):
            b = z.shape[0]
            return {"slot_logits": torch.zeros(b, 2), "brush_logit": torch.zeros(b)}

    cl.model = _M()
    lw, la, parts = cl.forward_g(torch.zeros(3, 4, 32, 32),
                                 torch.zeros(3, 1, 64, 64),
                                 torch.zeros(3, dtype=torch.long),
                                 torch.zeros(3, dtype=torch.long))
    assert isinstance(parts["writer"], torch.Tensor), "parts 必须是 tensor ✓（P1-6）"
    assert isinstance(parts["away"], torch.Tensor), "parts 必须是 tensor ✓（P1-6）"
    assert not parts["writer"].requires_grad, "parts 必须 detached ✓"


def test_set_char_lookup_uses_codepoint(tmp_path):
    """P1-7 ✓: 键 = (script_id, Unicode codepoint) ✓ 跨进程稳定 ✓"""
    import csv as _csv

    p = os.path.join(str(tmp_path), "t.csv")
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["script", "character"])
        w.writeheader()
        w.writerow({"script": "楷", "character": "永"})
    cl = _fake_critic(["楷·永"])
    cl.set_char_lookup(p, ["."], ["楷", "行"])
    assert (0, ord("永")) in cl._lut, cl._lut
    assert cl._lut[(0, ord("永"))] == 0
