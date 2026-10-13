#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_checkpoint.py — 一个入口，**显式判定**该走哪条条件分支，再交给对应评测器。

分支只有两条（详见 docs/inference_branches.md）:
  · 分支 T (table): ckpt 无 glyph 分支 -> g 必须是 None          -> tools/eval_table_checkpoint.py
  · 分支 S (std)  : ckpt 有 glyph 分支 -> g 必须是非空软掩码     -> tools/eval_std_soft_checkpoint.py

判定依据**只来自 ckpt 自己的 args**，并且会把依据打印出来（不做任何猜测、不静默兜底 ✗）。
判定不明（例如有 glyph 分支但不是 std_soft，而是旧骨架 latent）时**直接报错**，
要求调用者显式说明 —— 这正是 2026-10-12"评测把条件拔掉却照样出分"的反面做法。

用法:
  $PY tools/eval_checkpoint.py --ckpt <ckpt.pt> --out <新的空目录> \
      --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema
  # 想强制指定分支（一般不需要）: --branch std|table
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.eval.protocol import (EvalRequest, git_revision, is_complete,   # noqa: E402
                               sha256_file, sha256_tree, write_done, write_protocol)

VAE_EXTS = ("config.json", ".safetensors", ".bin", ".txt", ".model")   # HF VAE 目录 ✓


def build_request(ckpt_path, vae_path, args, *, sampler="heun", steps=50, cfg=1.0,
                  gamma=0.0, seed=0, ema=True, sets_spec=""):
    """从 ckpt + 参数构造**冻结协议**（审查 §4.3 ✓）。

    · VAE 缩放走 src/utils/vae_scaling 的**唯一真源** ✓ —— 未登记 VAE 会**报错**
      (fail-closed ✓), 绝不用 SD 默认值冒充 ✗。
    · 身份 = ckpt/VAE 的 sha256 + 采样协议 + 代码版本 ⇒ 任一变化都会得到**新 eval_id** ✓✓。
    """
    from types import SimpleNamespace
    from src.utils.vae_scaling import resolve_vae_scaling
    # ⚠ 顺序很重要 ✓: 先确认目录存在（避免 sha256_tree 抛出"目录不存在"掩盖真正的问题 ✗）
    if not os.path.isdir(vae_path):
        raise ValueError("[eval_checkpoint] --vae-path 不是目录: %r ✗（显式指定就必须存在 ✓）" % vae_path)
    _a = SimpleNamespace(vae_scaling_factor=args.get("vae_scaling_factor"),
                         vae_shift_factor=args.get("vae_shift_factor"), vae_path=vae_path)
    # 未登记 **且** 没有显式 sf ⇒ 抛 ValueError（fail-closed ✓）;
    # 有显式 sf 则用它 ✓（这是被认可的逃生路径 ✓ —— ckpt 自带 vae_scaling_factor 的情形）
    sf, sh = resolve_vae_scaling(_a, vae_path)
    return EvalRequest(
        checkpoint_sha256=sha256_file(ckpt_path),
        vae_sha256=sha256_tree(vae_path, exts=VAE_EXTS),
        vae_scale=float(sf), vae_shift=float(sh), sampler=str(sampler),
        num_steps=int(steps), cfg=float(cfg), gamma=float(gamma), seed=int(seed),
        ema=bool(ema), code_revision=git_revision(), sets_spec=str(sets_spec or ""),
        extra={"content_cond": args.get("content_cond"),
               "no_char_cond": bool(args.get("no_char_cond", False)),
               "std_soft_cond": bool(args.get("std_soft_cond", False))})



def ckpt_args(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    return vars(a) if hasattr(a, "__dict__") else (a or {})


def decide(args):
    """返回 (branch, 依据字符串)。不确定则抛错——绝不猜 ✗。"""
    std_soft = bool(args.get("std_soft_cond", False))
    skel = bool(args.get("skel_as_glyph_cond", False)) or float(args.get("w_glyph_cond", 0) or 0) > 0
    vec = bool(args.get("glyph_vec_cond", False))

    if std_soft and (skel or vec):
        raise SystemExit("[eval_checkpoint] 同时开了 std_soft 与旧字形条件，判定不明 ✗ 请用 --branch 显式指定")
    if std_soft:
        return "std", "std_soft_cond=True（有 std 字形分支: glyph_embedder 必须收到非空 (N,1,64,64)）"
    if skel or vec:
        raise SystemExit("[eval_checkpoint] 这是**旧骨架/向量字形**分支（skel_as_glyph_cond / w_glyph_cond / glyph_vec_cond），"
                         "本仓库暂无对应的严格评测器 ✗ 请勿用 table 分支去评它")
    return "table", "std_soft_cond 未开（训练时就没有字形分支 -> g 必须是 None，喂 g 才是 bug）"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="必须是新的空目录（禁止混协议）")
    ap.add_argument("--vae-path", required=True)
    ap.add_argument("--branch", choices=["std", "table"], default=None)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="即使该 eval_id 已完成 / 目录里是别的协议, 也强制重跑并覆盖 ✓")
    a = ap.parse_args()

    args = ckpt_args(a.ckpt)
    auto_branch, why = decide(args)
    branch = a.branch or auto_branch
    if a.branch and a.branch != auto_branch:
        print("[eval_checkpoint] ⚠ 手工指定 %s，但 ckpt 判定是 %s（%s）" % (a.branch, auto_branch, why))
    print("[eval_checkpoint] ckpt      = %s" % a.ckpt)
    print("[eval_checkpoint] 分支判定  = **%s**" % branch)
    print("[eval_checkpoint] 依据      = %s" % why)
    print("[eval_checkpoint] 协议      = gamma=%s steps=%s cfg=%s batch=%s sampler=heun(固定) 权重=ema"
          % (a.gamma, a.steps, a.cfg, a.batch))

    # ★ 2026-10-13 [审查 §4.3] **评测身份**: 先写不可变协议(protocol.json + eval_id) ✓
    req = build_request(a.ckpt, a.vae_path, args, sampler="heun", steps=a.steps,
                        cfg=a.cfg, gamma=a.gamma,
                        seed=int(args.get("global_seed", 0) or 0), ema=True,
                        sets_spec=str(a.sets or ""))
    eid = req.eval_id()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print("[eval_checkpoint] eval_id    = %s" % eid)

    # ★ 2026-10-13 [审查 §4.3 后半句] "同一 eval_id 可以**安全复用**" ✓✓
    #   并严格执行本工具自己的承诺"禁止混协议" ✓（原来只是**口头**承诺 ✗: 它会直接把
    #   目录里已有的 protocol.json **覆盖**掉, 从而抹掉"这目录里到底跑过什么"的记录 ✗✗）
    import json as _json
    _pp = Path(a.out) / "protocol.json"
    _prev = None
    if _pp.is_file():
        try:
            _prev = _json.loads(_pp.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            _prev = "unreadable"
    if _prev is not None and not a.force:
        if _prev == "unreadable":
            raise SystemExit("[eval_checkpoint] ✗ %s 存在但不可解析 ⇒ 拒绝在未知协议上重跑 ✗（换 --out 或 --force）" % _pp)
        if _prev.get("eval_id") != eid:
            raise SystemExit(
                "[eval_checkpoint] ✗ 该目录已属于**另一个协议**(eval_id=%s ≠ 本次 %s) ⇒ 禁止混协议 ✗。"
                "\n                   请换 --out；确要覆盖请显式加 --force ✓"
                % (str(_prev.get("eval_id"))[:12], eid[:12]))
    _ok, _why = is_complete(req, Path(a.out))
    if _ok and not a.force:
        print("[eval_checkpoint] ✓ 该 eval_id **已完成 ⇒ 安全复用, 不重跑** ✓（--force 可强制重跑）")
        print("[eval_checkpoint]   %s" % _why)
        return 0
    write_protocol(req, Path(a.out))
    print("[eval_checkpoint] 协议已写入 = %s" % _pp)


    tool = ROOT / ("tools/eval_std_soft_checkpoint.py" if branch == "std"
                   else "tools/eval_table_checkpoint.py")
    if not tool.is_file():
        raise SystemExit("[eval_checkpoint] 缺评测器: %s" % tool)

    cmd = [sys.executable, "-u", str(tool), "--ckpt", a.ckpt, "--out", a.out,
           "--gamma", str(a.gamma), "--steps", str(a.steps), "--cfg", str(a.cfg),
           "--batch", str(a.batch), "--device", a.device, "--vae-path", a.vae_path]
    if a.sets is not None:
        cmd += ["--sets", a.sets]
    if a.preflight_only:
        cmd += ["--preflight-only"]
    print("[eval_checkpoint] 调用      = %s" % " ".join(cmd[1:]))
    rc = subprocess.call(cmd)
    if rc == 0 and a.preflight_only:
        # ★ 真 bug 修复: 原先 --preflight-only 也 rc==0 ⇒ 会写 DONE.json = **假完成** ✗✗
        print("[eval_checkpoint] · --preflight-only ⇒ **不写 DONE**（只做预检, 未真正评测 ✓）")
        return 0
    if rc == 0:
        # ★ 只有**跑完**才写 DONE ✓（审查: "只有在所有样本、评分文件和哈希校验成功之后才写入" ✓✓）
        n_png = len(list(Path(a.out).rglob("*.png")))
        write_done(req, Path(a.out), n_samples=n_png)
        print("[eval_checkpoint] ✓ 完成: eval_id=%s, 样本 %d 张, DONE.json 已写 ✓" % (eid[:12], n_png))
    return rc


if __name__ == "__main__":
    sys.exit(main())
