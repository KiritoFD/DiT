#!/usr/bin/env python
"""诊断: (1) 各模块梯度占比 (2) g 条件消融 —— 回答"g 到底有没有被用"。

## 为什么
v15c 的 strict 在 155k 后平台（0.564），而 seen 涨到 0.696（gap 0.132）。
历史诊断说"g 注入作用仅 ~2.7%、glyph_scale 梯度近零" —— 需要复核：
  - 如果 g 的梯度/影响很小 -> 模型没在读 g，加数据也没用，要改注入方式
  - 如果 g 影响大 -> 问题在数据量/正则，加增强 + 开 dropout

## 用法（CPU 也可）
  PYTHONPATH=. python tools/diag_grad_and_g.py \
      --ckpt assets/results/v15c_fixed/<run>/checkpoints/0210000.pt \
      --steps 3
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/train/configs/v15c_fixed.json")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, default=3, help="梯度诊断跑几步")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    a = get_args()
    dev = torch.device(a.device)

    # ── 复用 train.py 的构建路径（避免手写字段漏项，doc68 D1 坑）──
    from src.train.cli import parse_args as _pa
    sys.argv = ["x", "--config", a.config]
    args = _pa()
    args.device = a.device
    args.cond_drop_all_prob = 0.0     # 诊断时关掉 dropout，看真实梯度
    args.cond_drop_one_prob = 0.0
    args.cond_drop_which_glyph_prob = 0.0

    from src.train.train import build_model, build_dataset  # noqa: E402
    print("[diag] 构建模型/数据集 ...", flush=True)
    model = build_model(args).to(dev)
    ds = build_dataset(args)

    # 加载 ckpt
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("model")
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    ms, us = model.load_state_dict(sd, strict=False)
    print(f"[diag] ckpt 加载: missing={len(ms)} unexpected={len(us)}", flush=True)
    model.train()

    # ── 梯度诊断 ─────────────────────────────────────────────────────
    groups = {
        "blocks": [], "glyph_embedder": [], "inj_out_proj": [],
        "callig_embedder": [], "char_embedder": [], "final_layer": [],
        "cond_fusion": [], "other": [],
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        hit = False
        for k in groups:
            if k == "other":
                continue
            if k in n:
                groups[k].append(p)
                hit = True
                break
        if not hit:
            groups["other"].append(p)

    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True,
                                         num_workers=0)
    it = iter(loader)
    from src.loss.repa import RepaModule  # noqa: F401  (确认可 import)

    print(f"\n[diag] 跑 {a.steps} 步看梯度 ...", flush=True)
    for step in range(a.steps):
        batch = next(it)
        loss = _one_step(model, args, batch, dev)
        loss.backward()
        if step == 0:
            print(f"\n  === 各模块梯度范数（step {step}）===")
            tot = 0.0
            norms = {}
            for k, ps in groups.items():
                s = 0.0
                for p in ps:
                    if p.grad is not None:
                        s += float(p.grad.detach().norm() ** 2)
                norms[k] = s ** 0.5
                tot += s
            tot = tot ** 0.5
            for k, v in sorted(norms.items(), key=lambda x: -x[1]):
                if not groups[k]:
                    continue
                print(f"    {k:<18} {v:>10.4f}   ({v/max(tot,1e-12)*100:>5.1f}%)")
            print(f"    {'总':<18} {tot:>10.4f}")
        model.zero_grad(set_to_none=True)

    print("\n[diag] 完成（g 消融需要 eval 路径，见 tools/diag_g_ablation.py）")


def _one_step(model, args, batch, dev):
    """极简一步：把 batch 拆开喂模型，算 flow/DDPM loss。"""
    if isinstance(batch, dict):
        x = batch["x"].to(dev)
        t = batch.get("t")
        if t is None:
            t = torch.rand(x.shape[0], device=dev)
        else:
            t = t.to(dev)
        out = model(x, t, **{k: v.to(dev) for k, v in batch.items()
                             if k not in ("x", "t") and torch.is_tensor(v)})
    else:
        raise RuntimeError(f"未知 batch 类型: {type(batch)}")
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.float().pow(2).mean()


if __name__ == "__main__":
    main()
