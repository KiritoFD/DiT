# -*- coding: utf-8 -*-
"""
muon.py — Muon 优化器 (极分解正交化 + AdamW 混合)。

Muon (Keller Jordan / Moonlight 2025): 2D 矩阵权重用 momentum + 极分解
正交化 (0 阶谱更新), 其余参数 (标量/向量/embedding) 走 AdamW。对
Transformer 这类"矩阵权重占绝对多数"的模型, 通常带来 1.3-1.5x 样本效率。

正交化实现: 用 SVD 极分解 (P = U V^T, 精确 ~1e-7), 对本项目 384 维权重
每步开销 ~10-50ms (可忽略 vs 3 step/s 训练)。同时也提供 NS5 迭代版本
(zeropower_via_newtonschulz5, Moonlight 风格, 更快但需谱预归一)。

用法 (兼容 torch.optim.AdamW 的 param_groups 语义):
    from src.optim.muon import Muon
    opt = Muon(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.02, weight_decay=0.0,          # muon 组 (矩阵)
        adamw_lr=3e-4, adamw_weight_decay=0.01,  # adamw 组 (向量/embedding)
    )
    # 也可显式 param_groups: 组内 "lr"<=0.1 或 "muon":False → AdamW
"""
import torch
from torch.optim import Optimizer


def _polar_svd(G: torch.Tensor) -> torch.Tensor:
    """SVD 极分解: G → P = U V^T (酉因子)。精确稳定 (~1e-7)."""
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 20) -> torch.Tensor:
    """Newton-Schulz 迭代逼近极分解 (Moonlight 风格, 需谱预归一才收敛)。

    对 G(n,m) n<=m: 对 S = G@G.T (n,n) 迭代 NS, 得正交 U, 再 W = U@(U.T@G)。
    注意: 需先谱归一 S 到 σ1≈1 (默认 Frobenius 归一, steps 需较多才精确)。
    用 SVD 版 (默认) 更稳; 此函数仅供实验/性能对比。
    """
    a, b, c = (1.5, -0.5, 0.0)  # Newton 法系数 X(3I - X^T X)/2 (二次收敛)
    U = G
    if U.size(0) > U.size(1):
        U = U.T
    S = U @ U.T  # (n,n)
    S = S / (S.norm() + 1e-8)
    for _ in range(steps):
        S = S @ (a * torch.eye(S.size(0), device=S.device, dtype=S.dtype) - b * (S.T @ S))
        S = S / (S.norm() + 1e-8)
    W = S @ (S.T @ U)
    if G.size(0) > G.size(1):
        W = W.T
    return W


class Muon(Optimizer):
    """矩阵 → 极分解正交 momentum 更新; 非矩阵 → AdamW。

    param_groups 语义:
      - "muon":False 或 "lr"<=0.1 且未显式标 muon: 该组走 AdamW
      - 否则 (默认, dim>=2 自动组): 走 Muon 矩阵更新
    简洁用法: Muon(list_of_params, lr=0.02, adamw_lr=3e-4) 自动分组。
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=20,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 adamw_lr=3e-4, adamw_weight_decay=0.01, use_svd=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        auto_groups = None
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], torch.Tensor):
            mat = [p for p in params if p.dim() >= 2]
            vec = [p for p in params if p.dim() < 2]
            auto_groups = [
                {"params": mat, "lr": lr, "weight_decay": weight_decay, "muon": True},
                {"params": vec, "lr": adamw_lr, "weight_decay": adamw_weight_decay, "muon": False},
            ]
        defaults = dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        betas=betas, eps=eps, use_svd=use_svd)
        super().__init__(auto_groups or params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            muon = group.get("muon", True)
            lr = group.get("lr", 0.02)
            is_adamw = (not muon) or lr <= 0.1
            wd = group.get("weight_decay", 0.0)
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            betas = group["betas"]
            eps = group["eps"]
            use_svd = group.get("use_svd", True)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                if is_adamw:
                    b1, b2 = betas
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    step = state["step"] + 1
                    state["step"] = step
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bias_c1 = 1 - b1 ** step
                    bias_c2 = 1 - b2 ** step
                    step_size = lr / bias_c1
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
                else:
                    # Muon 矩阵更新: momentum → 极分解正交 → 更新
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                    if use_svd:
                        upd = _polar_svd(g_eff)
                    else:
                        upd = zeropower_via_newtonschulz5(g_eff, ns_steps)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr)
        return loss