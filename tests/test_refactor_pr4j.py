# -*- coding: utf-8 -*-
"""test_refactor_pr4j.py —— 2026-10-14 [重构 PR4] **单步输入解析** 的验收测试 ✓

审查 §4.2 第 1 条: "H2D 数据移动" ✓

本组测试锁定:
  ① 三分支的形状/数值与旧内联**逐字一致** ✓
  ② ★★ **三分支的梯度语义不同** ✗✗ —— 这是搬移时最容易被"顺手统一"而改坏的地方 ✓
       ① `train_vae_enc`: 用 `rsample()` ✓ **可导** ✓（要训 VAE ✓）
       ② `latent` 缓存  : 纯 H2D ✓ 无 VAE ✓
       ③ 实时图像       : `sample()` + `no_grad` + fp32 autocast ✓ **不可导** ✓
     ⇒ 本模块**刻意不统一** ✓ 用测试钉死 ✓
  ③ latent 合同: `(z - SHIFT) * SF` ✓（合同 I2 ✓ 两条路同约定 ✓）
"""
from __future__ import annotations

import io
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.step import resolve_step_inputs, StepInputs                # noqa: E402


# ─────────────── 假 VAE（可观测 rsample / sample 是否被调用 ✓）───────────────
class _Dist:
    def __init__(self, z, log=0.0):
        self._z = z
        self.logvar = torch.full_like(z, log)
        self.mean = z

    def rsample(self):
        return self._z.clone()

    def sample(self):
        return self._z.clone()


class _VAE:
    def __init__(self, z):
        self.z = z
        self.rsample_calls = 0
        self.sample_calls = 0

    def encode(self, x):
        z = self.z
        vae = self

        class _P:
            @property
            def latent_dist(self):
                return _Dist(z)

        return _P()


class _VAEPlain:
    def __init__(self, z):
        self.z = z

    def encode(self, x):
        z = self.z

        class _P:
            latent_dist = _Dist(z)

        return _P()


# ─────────────── ② latent 缓存分支 ───────────────
def test_latent_branch_basic():
    b = {"latent": torch.randn(2, 4, 8, 8)}
    r = resolve_step_inputs(b, None, "cpu", train_vae_enc=False,
                            vae_sf=0.18215, vae_shift=0.0)
    assert isinstance(r, StepInputs)
    assert tuple(r.x_latent.shape) == (2, 4, 8, 8)
    assert r.x is None and r.loss_vae_kl is None
    assert torch.equal(r.x_latent, b["latent"]), "latent 分支不得改动数值 ✓"


def test_latent_branch_concats_aux():
    b = {"latent": torch.randn(2, 4, 8, 8), "aux_latents": torch.randn(2, 8, 8, 8)}
    r = resolve_step_inputs(b, None, "cpu", train_vae_enc=False,
                            vae_sf=1.0, vae_shift=0.0)
    assert tuple(r.x_latent.shape) == (2, 12, 8, 8), "aux 必须拼在通道维 ✓"
    assert torch.equal(r.x_latent[:, :4], b["latent"])
    assert torch.equal(r.x_latent[:, 4:], b["aux_latents"])


def test_latent_branch_skips_empty_aux():
    b = {"latent": torch.randn(2, 4, 8, 8),
         "aux_latents": torch.zeros(0)}          # numel()==0 ⇒ 不拼 ✓
    r = resolve_step_inputs(b, None, "cpu", train_vae_enc=False,
                            vae_sf=1.0, vae_shift=0.0)
    assert tuple(r.x_latent.shape) == (2, 4, 8, 8)


def test_latent_branch_passes_image_through():
    b = {"latent": torch.randn(1, 4, 4, 4), "image": torch.randn(1, 3, 16, 16)}
    r = resolve_step_inputs(b, None, "cpu", train_vae_enc=False,
                            vae_sf=1.0, vae_shift=0.0)
    assert r.x is not None and torch.equal(r.x, b["image"])


# ─────────────── ③ 实时图像分支（不可导 ✓）───────────────
def test_image_branch_uint8_scaling_and_latent_contract():
    """★ latent 合同: `(z - SHIFT) * SF` ✓（合同 I2 ✓ 两条路同约定 ✓）"""
    z = torch.full((1, 4, 4, 4), 0.75)
    vae = _VAEPlain(z)
    b = {"image": torch.zeros(1, 3, 16, 16, dtype=torch.uint8)}
    r = resolve_step_inputs(b, vae, "cpu", train_vae_enc=False,
                            vae_sf=2.0, vae_shift=0.5)
    assert torch.allclose(r.x_latent, (z - 0.5) * 2.0), "必须 (z-SHIFT)*SF ✓"
    assert r.loss_vae_kl is None
    # uint8 图必须被映射到 [-1,1] ✓（0 ⇒ -1 ✓）
    assert torch.allclose(r.x, torch.full_like(r.x, -1.0)), "uint8 0 ⇒ -1 ✓"


def test_image_branch_is_not_differentiable():
    """★★ 该分支**刻意** `no_grad` + `sample()` ✓ ⇒ 不建图 ✓（要训 VAE 时走另一分支 ✓）"""
    z = torch.randn(1, 4, 4, 4, requires_grad=True)
    vae = _VAEPlain(z)
    b = {"image": torch.zeros(1, 3, 16, 16, dtype=torch.uint8)}
    r = resolve_step_inputs(b, vae, "cpu", train_vae_enc=False,
                            vae_sf=1.0, vae_shift=0.0)
    assert not r.x_latent.requires_grad, "图像分支不得建图 ✗（no_grad ✓）"
    assert r.x_latent.grad_fn is None


# ─────────────── ① vae-enc 分支（可导 ✓ 与上面**不同** ✓）───────────────
def test_vae_enc_branch_is_differentiable_and_uses_rsample():
    """★★ 与图像分支**相反** ✓: 这里必须 `rsample()` 且**可导** ✓（要训 VAE Encoder ✓）"""
    z = torch.randn(2, 4, 4, 4, requires_grad=True)
    vae = _VAEPlain(z)
    b = {"image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8)}
    r = resolve_step_inputs(b, vae, "cpu", train_vae_enc=True,
                            vae_sf=1.0, vae_shift=0.0)
    assert r.x_latent.requires_grad, "vae-enc 分支必须**可导** ✗（要训 VAE ✓）"
    assert r.loss_vae_kl is not None and r.loss_vae_kl.requires_grad, "KL 必须可导 ✓"
    # 梯度能流回 z ✓
    r.x_latent.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0, "梯度必须流回 z ✓"


def test_vae_enc_branch_computes_kl():
    z = torch.zeros(2, 4, 4, 4)
    vae = _VAEPlain(z)
    b = {"image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8)}
    r = resolve_step_inputs(b, vae, "cpu", train_vae_enc=True,
                            vae_sf=1.0, vae_shift=0.0)
    # logvar=0, mean=0 ⇒ KL = -0.5 * sum(1 + 0 - 0 - 1) = 0 ✓
    assert abs(float(r.loss_vae_kl)) < 1e-6, float(r.loss_vae_kl)


def test_vae_enc_requires_image_key():
    """`train_vae_enc=True` 但 batch 无 image ⇒ 回退到 latent 分支 ✓（与旧内联一致 ✓）"""
    b = {"latent": torch.randn(1, 4, 4, 4)}
    r = resolve_step_inputs(b, None, "cpu", train_vae_enc=True,
                            vae_sf=1.0, vae_shift=0.0)
    assert r.loss_vae_kl is None and tuple(r.x_latent.shape) == (1, 4, 4, 4)


# ─────────────── 接线守卫 ───────────────
def test_train_py_uses_resolve_step_inputs():
    t = io.open("src/train/train.py", encoding="utf-8").read()
    assert "_resolve_step_inputs(batch, vae, device," in t, "未接线 ✗"
    assert "x_latent, x, loss_vae_kl = _si.x_latent, _si.x, _si.loss_vae_kl" in t, \
        "下游变量名必须保持 ✓"
    # 旧内联的 rsample / posterior 代码必须已消失 ✓
    assert "posterior = vae.encode(x).latent_dist" not in t, "旧内联 VAE 编码仍在 ✗"
    assert "posterior.rsample()" not in t, "旧内联 rsample 仍在 ✗"
