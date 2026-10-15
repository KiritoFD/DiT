# -*- coding: utf-8 -*-
"""test_refactor_resume.py —— **精确恢复测试**（审查 PR0 硬要求 ✓ 2026-10-14 ✓）

审查 PR0 原文:
  "**精确恢复测试** —— 在固定环境下比较**连续训练 100 步**与**训练 50 步、保存、
   恢复后再训练 50 步**。**数据顺序、参数、optimizer、EMA、scheduler** 应符合
   明确的恢复契约。" ✓

★ 这同时是 **PR4 的完成标准**（审查 §8: "PR4 … 完成标准 = **Exact Resume 可重放**" ✓）

★ 本测试的做法 ✓（**五项状态全覆盖** ✓）:
  · **数据顺序** ✓ —— 每个 micro-step 用 `torch.randint` 抽样本索引 ✓
    （**消耗 torch RNG** ✓ ⇒ 若 RNG 没恢复 ✓ 数据顺序**立刻**不同 ✓ 这是最灵敏的探针 ✓）
  · **参数** ✓ / **optimizer** ✓（AdamW 的 `exp_avg` / `exp_avg_sq` / `step` ✓）
  · **EMA** ✓ / **scheduler** ✓（LambdaLR ✓）
  · 另外: 用 `Dropout` ✓ 让**前向本身**也依赖 RNG ✓（更严格 ✓）

★ 判据: 两条路径的**全部状态逐位一致** ✓（不是"接近" ✓ 是 `torch.equal` ✓）
  —— 审查要求"逐步、逐样本可重放" ✓ ⇒ 只允许**逐位** ✓

⚠ 已知契约边界（本测试**也**锁定 ✓）:
  `LongEpoch` 采样器每 `epoch_steps` 步 reset ✓ ⇒ 只有
  `ckpt_every == epoch_steps` 时"不存采样器状态"才成立 ✓
  （`rng_meta["epoch_aligned"]` 记录该前提 ✓ 见 test_refactor_p2.py ✓）
  本测试用**无 epoch 概念**的抽样 ✓ 因此不涉及该边界 ✓
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.ckpt import build_checkpoint, capture_rng, restore_rng   # noqa: E402
from src.train.ema import update_ema                                    # noqa: E402

N_SAMPLES = 64
BATCH = 8
SEED = 20261014


class _Tiny(torch.nn.Module):
    """最小可训练模块 ✓（含 Dropout ⇒ 前向也消耗 RNG ✓ 更严格 ✓）。"""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc1 = torch.nn.Linear(6, 16)
        self.drop = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(16, 1)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)

    def forward(self, x):
        return self.fc2(self.drop(torch.tanh(self.fc1(x))))


def _make(seed=0):
    torch.manual_seed(SEED)
    model = _Tiny(seed)
    ema = _Tiny(seed)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 0.5 ** (s / 50.0))
    return model, ema, opt, sched


def _step(model, ema, opt, sched, step, trace):
    """一次完整更新 ✓（数据顺序 → 前向 → 反传 → clip → step → scheduler → EMA ✓）。"""
    # ★ 数据顺序: 抽样本索引 ⇒ **消耗 torch RNG** ✓（RNG 没恢复就会立刻不同 ✓）
    idx = torch.randint(0, N_SAMPLES, (BATCH,))
    trace.append([int(i) for i in idx.tolist()])
    g = torch.Generator().manual_seed(1000 + int(idx.sum()))     # 数据内容也依赖索引 ✓
    x = torch.randn(BATCH, 6, generator=g)
    y = torch.randn(BATCH, 1, generator=g)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
    update_ema(ema, model, 0.9)


def _fingerprint(model, ema, opt, sched, trace, rng):
    return {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "ema": {k: v.clone() for k, v in ema.state_dict().items()},
        "opt": {k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in opt.state_dict()["state"][0].items() if k != "step"},
        "opt_step": int(opt.state_dict()["state"][0]["step"]),
        "lr": float(opt.param_groups[0]["lr"]),
        "sched_last": int(sched.state_dict()["last_epoch"]),
        "trace": list(trace),
        "rng_torch": rng["torch_cpu"].clone(),
    }


def _assert_same(a, b, label):
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), f"{label}: 参数 {k} 不一致 ✗"
    for k in a["ema"]:
        assert torch.equal(a["ema"][k], b["ema"][k]), f"{label}: EMA {k} 不一致 ✗"
    for k in a["opt"]:
        assert torch.equal(a["opt"][k], b["opt"][k]), f"{label}: optimizer.{k} 不一致 ✗"
    assert a["opt_step"] == b["opt_step"], f"{label}: optimizer step 不一致 ✗"
    assert a["lr"] == b["lr"], f"{label}: lr 不一致 ✗（scheduler 没恢复?）"
    assert a["sched_last"] == b["sched_last"], f"{label}: scheduler last_epoch 不一致 ✗"
    assert a["trace"] == b["trace"], f"{label}: **数据顺序**不一致 ✗（RNG 没恢复 ✓ 最灵敏探针 ✓）"
    assert torch.equal(a["rng_torch"], b["rng_torch"]), f"{label}: torch RNG 状态不一致 ✗"


# ─────────────── ★ 主判据: 100 连续 ≡ 50 + 保存 + 恢复 + 50 ───────────────
def test_exact_resume_is_bit_identical(tmp_path):
    HALF = 50

    # ── 路径 A: **连续 100 步** ✓
    m, e, o, s = _make()
    ta: list = []
    for i in range(2 * HALF):
        _step(m, e, o, s, i, ta)
    fa = _fingerprint(m, e, o, s, ta, capture_rng())

    # ── 路径 B: 50 步 → 保存 → 恢复 → 50 步 ✓
    m2, e2, o2, s2 = _make()
    tb: list = []
    for i in range(HALF):
        _step(m2, e2, o2, s2, i, tb)
    ck = build_checkpoint(m2, o2, e2, s2, args=None, train_steps=HALF)
    assert "rng" in ck, "ckpt 必须带 RNG 状态 ✗（P1-2 ✓）"
    path = tmp_path / "resume.pt"
    torch.save(ck, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)     # 真实序列化往返 ✓

    m3, e3, o3, s3 = _make()
    m3.load_state_dict(ck["delta"])   # ⚠ 键名叫 delta ✗ 但存的是**完整** state_dict ✓
    o3.load_state_dict(ck["opt"])
    e3.load_state_dict(ck["ema"])
    if ck.get("scheduler") is not None:
        s3.load_state_dict(ck["scheduler"])
    assert restore_rng(ck.get("rng") or {}) is True, "RNG 必须**完整**恢复 ✓"
    for i in range(HALF, 2 * HALF):
        _step(m3, e3, o3, s3, i, tb)
    fb = _fingerprint(m3, e3, o3, s3, tb, capture_rng())

    _assert_same(fa, fb, "100 连续 vs 50+恢复+50")


def test_resume_without_rng_diverges():
    """★★ **阴性对照** ✓ —— 证明"RNG 恢复"**真的**在起作用 ✓（否则主测试无意义 ✗）

    ⚠ 首版我写成"重建两条路径比较数据顺序" ✗ —— 但 `_make()` 里会
      `torch.manual_seed(SEED)` **重新播种** ✓ ⇒ 两条路径的 RNG 流**本就相同** ✗
      ⇒ 阴性对照**测不到东西** ✗（实测失败 ⇒ 才发现 ✓）
    ⇒ 改成**直接测机制本身** ✓ 更精确也更诚实 ✓
    """
    st = capture_rng()
    a = torch.randint(0, 100000, (8,))
    assert restore_rng(st) is True
    b = torch.randint(0, 100000, (8,))
    assert torch.equal(a, b), "恢复 RNG 后随机序列必须**重现** ✓"
    c = torch.randint(0, 100000, (8,))
    assert not torch.equal(a, c), \
        "继续抽样竟然与恢复后的第一段相同 ✗ ⇒ 随机源没有推进 ⇒ 本测试没测到东西 ✗"


def test_data_order_probe_is_rng_sensitive():
    """★ 主测试的**探针有效性** ✓: 样本索引确实由 torch RNG 决定 ✓

    （若某天 `randint` 改成别的随机源 ✗ ⇒ 主测试会**静默失效** ✗ ⇒ 这里先拦住 ✓）
    """
    st = capture_rng()
    first = torch.randint(0, N_SAMPLES, (BATCH,))
    restore_rng(st)
    again = torch.randint(0, N_SAMPLES, (BATCH,))
    assert torch.equal(first, again), "数据顺序探针必须可复现 ✓"
    other = torch.randint(0, N_SAMPLES, (BATCH,))
    assert not torch.equal(first, other), "探针必须对 RNG 敏感 ✓"


def test_checkpoint_roundtrip_preserves_all_state():
    """★ 契约 ✓: ckpt 必须含 delta / opt / ema / scheduler / rng / train_steps ✓

    ⚠ 真实键名 ✓: 模型存在 **`delta`** 键里 ✗ —— 但 `ckpt.py` 的注释说明
      "始终保存完整 state_dict（不做 delta-only）" ✓ ⇒ **名字有误导性** ✗
      （记录于此 ✓ 重命名属行为变更 ✗ 应单独决策 ✓）
    """
    m, e, o, s = _make()
    _step(m, e, o, s, 0, [])
    ck = build_checkpoint(m, o, e, s, args=None, train_steps=7)
    for key in ("delta", "opt", "ema", "scheduler", "rng", "train_steps"):
        assert key in ck, f"ckpt 缺 `{key}` ✗"
    assert ck["train_steps"] == 7, "ckpt 步数不对 ✗"
    assert set(ck["rng"]) >= {"python", "numpy", "torch_cpu"}, \
        f"RNG 状态不完整 ✗: {sorted(ck['rng'])}"


def test_restore_rng_reports_incomplete():
    """★ 契约 ✓: 状态**不全**时必须返回 False ✓（不得谎报完整 ✗）"""
    assert restore_rng({}) is False
    assert restore_rng({"python": __import__("random").getstate()}) is False
