# -*- coding: utf-8 -*-
"""verify_grad_parity.py —— **梯度等价黄金**（审查 PR0 要求 ✓ 2026-10-14 ✓）

审查 PR0 原文:
  "**梯度等价测试** —— 在固定 FP32 小 batch 上，检查**主干和条件层**的梯度，
   避免'前向看起来正常，实际训练方向变了'。" ✓

★ 为什么必须单独做梯度黄金 ✗:
  已有的 `verify_forward_parity.py` 是 `torch.no_grad()` ✓ ⇒ 只锁**前向** ✓
  而 PR4 的搬移大量发生在**反传与更新**路径上 ✓:
    · backward 策略（joint / sequential / probe ✓）
    · 损失装配顺序（`ObjectiveComposer` ✓ **注册顺序 = 求和顺序 = 语义** ✗）
    · 梯度探针 / clip / EMA ✓
  ⇒ 前向黄金**管不到**这些 ✗ ⇒ 需要一份**梯度**黄金 ✓

★ 判据（本工具记录 ✓）:
  ① `grad_sha256`  —— 全参数梯度按**固定顺序**拼成一维后的 sha256 ✓（**逐位** ✓）
  ② `group_norms`  —— 按**参数角色分组**的梯度范数 ✓（主干 / 条件 / 头 ✓ 便于定位 ✓）
  ③ `loss`         —— 固定函数值 ✓（前向+反传一致性的旁证 ✓）

★ 用法::

    # 冻结黄金（重构前跑一次 ✓）
    python tools/verify_grad_parity.py --ckpt _local_ckpt/0055000.pt --save
    # 校验（重构后 ✓ 必须逐位一致 ✓）
    python tools/verify_grad_parity.py --ckpt _local_ckpt/0055000.pt --check

★ 设计取舍 ✓: 用**固定随机泛函** `(out * W).sum()` 作为损失 ✓
  （W 由固定 seed 生成 ✓）而不是完整 FM 损失 ✓ —— 这样黄金**不依赖**
  diffusion / REPA / 数据集的内部实现 ✓ ⇒ 它们各自的搬移不会误报 ✓
  但模型自身的**前向 + 反传 + 条件层梯度**仍被逐位锁定 ✓（这正是 PR4/PR5 要保护的 ✓）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.verify_forward_parity import load, make_inputs          # noqa: E402


# ── 参数角色分组（用于把"哪里变了"直接定位出来 ✓）──
_GROUPS = [
    ("backbone", ("blocks.", "x_embedder", "pos_embed", "final_layer", "t_embedder")),
    ("condition", ("y_embedder", "callig", "script", "char_", "glyph", "style_",
                   "writer", "pair", "std_", "cond_")),
    ("other", ()),
]


def group_of(name: str) -> str:
    for gname, keys in _GROUPS:
        if any(k in name for k in keys):
            return gname
    return "other"


def grad_fingerprint(model, loss) -> dict:
    """反传 + 记录**逐位**指纹 ✓。"""
    loss.backward()
    flat = []
    group_sq = {}
    n_with_grad = 0
    for name, p in model.named_parameters():
        g = p.grad
        gname = group_of(name)
        if g is None:
            continue
        n_with_grad += 1
        gf = g.detach().float().reshape(-1)
        flat.append(gf)
        group_sq[gname] = group_sq.get(gname, 0.0) + float(gf.pow(2).sum())
    if not flat:
        raise RuntimeError("没有任何参数拿到梯度 ✗（模型未参与计算?）")
    cat = torch.cat(flat).contiguous()
    buf = io.BytesIO()
    torch.save(cat.cpu(), buf)
    return {
        "grad_sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
        "n_params_with_grad": n_with_grad,
        "n_grad_elements": int(cat.numel()),
        "group_norms": {k: v ** 0.5 for k, v in sorted(group_sq.items())},
        "loss": float(loss.detach()),
    }


def compute(ckpt: str, seed: int = 1234, batch: int = 2, fseed: int = 4321) -> dict:
    """在固定 FP32 小 batch 上算梯度指纹 ✓（**确定性** ✓）。"""
    torch.manual_seed(fseed)                    # 固定 dropout / 条件 drop ✓
    model, a = load(ckpt, device="cpu")
    model.train()                               # ⚠ 必须 train ✓（dropout 是训练语义 ✓）
    x, t, yc, ych, kw = make_inputs(a, model, seed=seed, batch=batch)
    torch.manual_seed(fseed)                    # forward 前再播一次 ⇒ 逐位可复现 ✓
    out = model(x, t, yc, ych, **kw)
    out = out[0] if isinstance(out, tuple) else out
    # 固定随机泛函 ✓（W 由 fseed 生成 ✓ 与模型无关 ✓）
    gw = torch.Generator().manual_seed(fseed)
    W = torch.randn(out.shape, generator=gw)
    loss = (out.float() * W).sum()
    res = grad_fingerprint(model, loss)
    res.update({"ckpt": os.path.basename(ckpt), "seed": seed, "batch": batch,
                "fseed": fseed, "torch": torch.__version__})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", default="assets/_audit/grad_v84A55k.json")
    ap.add_argument("--save", action="store_true", help="写黄金（重构前冻结 ✓）")
    ap.add_argument("--check", action="store_true", help="校验（重构后 ✓）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    res = compute(args.ckpt, seed=args.seed, batch=args.batch)
    print(f"[grad] n_with_grad={res['n_params_with_grad']} "
          f"n_elem={res['n_grad_elements']} loss={res['loss']:.6f}")
    for k, v in res["group_norms"].items():
        print(f"[grad]   {k:10s} ‖g‖ = {v:.6e}")
    print(f"[grad] sha256 = {res['grad_sha256']}")

    if args.save:
        os.makedirs(os.path.dirname(args.golden) or ".", exist_ok=True)
        with open(args.golden, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, sort_keys=True)
        print(f"[grad] ✓ 黄金已写 {args.golden}")
        return 0

    if args.check:
        if not os.path.exists(args.golden):
            print(f"[grad] ✗ 黄金不存在: {args.golden}")
            return 2
        with open(args.golden, encoding="utf-8") as f:
            gold = json.load(f)
        if res["grad_sha256"] == gold["grad_sha256"]:
            print(f"[grad] ✓ **逐位一致**（{res['n_grad_elements']} 个梯度元素 ✓）")
            return 0
        print("[grad] ✗ 梯度**不一致** ✗ —— 训练方向已变 ✗")
        print(f"[grad]   golden {gold['grad_sha256']}")
        print(f"[grad]   now    {res['grad_sha256']}")
        for k in sorted(set(gold.get("group_norms", {})) | set(res["group_norms"])):
            g0 = gold.get("group_norms", {}).get(k)
            g1 = res["group_norms"].get(k)
            if g0 is None or g1 is None or abs(g0 - g1) > 1e-12 * max(1.0, abs(g0)):
                print(f"[grad]   {k}: {g0} -> {g1}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
