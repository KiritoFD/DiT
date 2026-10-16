# -*- coding: utf-8 -*-
"""评测采样器的 **batch 不变性** + **尾批 padding** 测试（2026-10-16 ✓ 纯 CPU ✓）

★★ 依据（用户指示"让评测尽可能快" ✓ 改了 `sample_latents` 的批循环 ✓）:
  ① 删掉每批一次的 `torch.cuda.empty_cache()` ✗（强制同步 + 释放训练侧缓存块 ✗✗）
  ② 尾批 pad 到固定 shape ✓（torch.compile 只见一种 shape ✓ 不再为 187%8=3 重编译 ✗）
  ③ 标签张量预建一次 ✓

★ 本测试在 **CPU** 上证明两件事（无需 GPU ✓）:
  ① **批不变性** ✓: 同一批样本，用不同 `batch` 切分采样 ⇒ 结果一致 ✓
     （FM 各样本独立 ✓ LayerNorm ✓ 无跨样本耦合 ✓ ⇒ 这是数学性质 ✓）
     ⇒ 改大 `in_mem_eval_batch`（8 → 64 ✓）**不改变评测结论** ✓
  ② **pad 正确性** ✓: 尾批被 pad 后，pad 部分的输出**必须被丢弃** ✓
     ⇒ 返回长度恒等于样本数 ✓ 且非尾样本与不 pad 时一致 ✓

★ 容差说明（如实 ✓）: 不同 batch 的 GEMM 归约顺序可能不同 ✗
  ⇒ 用 `allclose(rtol=1e-4, atol=1e-5)` ✓ 而非逐位 ✓（分布等价 ✓ 位级可能微差 ✓）
"""
import torch

from src.eval.sampler import sample_latents
from src.loss.flow_matching import FlowMatching


class _TinyModel(torch.nn.Module):
    """极小 DiT 替身 ✓ —— 只需接受 (x, t, **kwargs) 并返回同形张量 ✓。"""

    def __init__(self, ch=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(ch, 8, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(8, ch, 3, padding=1))

    def forward(self, x, t, **kw):
        return self.net(x)


def _mk_case(n=11, ch=4, size=8):
    """n=11 ⇒ batch=8 时尾批只有 3 个 ✓ 正好触发 padding 分支 ✓。"""
    torch.manual_seed(0)
    noise = torch.randn(n, ch, size, size)
    conds = [(i % 3, i % 5) for i in range(n)]
    return noise, conds


def _run(batch, n=11):
    noise, conds = _mk_case(n)
    fm = FlowMatching(num_steps=6, sampler="euler")
    model = _TinyModel()
    with torch.no_grad():
        return sample_latents(model, fm, noise, conds, cfg_scale=0.0,
                              batch=batch, device="cpu", seed=0)


def test_batch_invariance_cpu():
    """★ 不同 batch 切分 ⇒ 同一结果 ✓（allclose ✓ 容差见模块说明 ✓）。"""
    a = _run(8)
    b = _run(64)
    assert a.shape == b.shape, "shape 不一致 ✗ %s vs %s" % (a.shape, b.shape)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), (
        "batch 不变性被破坏 ✗ max|Δ|=%.3e\n"
        "⇒ sample_latents 的批循环改动改变了数值 ✗（应只是切分方式 ✓）"
        % (a - b).abs().max())


def test_tail_padding_discards_pad_outputs():
    """★ 尾批 pad 后：① 返回长度 = 样本数 ✓ ② 与不 pad 的整除情形一致性 ✓。"""
    out = _run(8, n=11)                     # 8 + 3 ⇒ 尾批 3 被 pad 到 8 ✓
    assert out.shape[0] == 11, "返回长度应为样本数 11 ✓ 实际 %s ✗" % (out.shape[0],)
    # 与 batch=11（整除、无 pad ✓）对比 ⇒ 应一致 ✓（同一噪声同一模型 ✓）
    ref = _run(11, n=11)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5), (
        "pad 路径与整除路径结果不一致 ✗ max|Δ|=%.3e" % (out - ref).abs().max())


def test_single_sample_batch_still_works():
    """★ batch=1（逐样本 ✓ 最慢的极端 ✓）也应与其它 batch 一致 ✓。"""
    a = _run(1)
    b = _run(4)
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), (
        "batch=1 与 batch=4 结果不一致 ✗ max|Δ|=%.3e" % (a - b).abs().max())
