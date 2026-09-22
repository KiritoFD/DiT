# -*- coding: utf-8 -*-
"""T0 — 表在动吗？（表权重随训练步的轨迹）

对齐 `60_diagnostics.md` §4 的 T0：确认可训表（E_callig / E_pair / y_callig_embedder）
到底有没有在动，动多少、什么时候动。

相比"终值 vs 初始化"，这里用**训练轨迹**：
    * `Δ‖W_step − W_0‖ / ‖W_0‖`    —— 相对位移（表动没动）
    * `cos(W_step, W_0)`（逐行平均）—— 方向是否改变
    * 行间余弦 / 行范数           —— 表是否塌缩（所有行变得一样）

⚠ 已知会"假不动"的两个原因（写进结论时必须一起说）：
   1. AdamW 的 weight decay **绕过** grad hook → 看 grad 会以为没动
   2. zero-init 的**一步延迟** → 第 1 步一定不动

用法:
    python tools/probe_table_motion.py \
        --ckpt-dir assets/results/v15b_supcon/<ts>/checkpoints \
        --key y_callig_embedder --out assets/t0_v15b_supcon.json
"""
import argparse
import glob
import json
import os
import re
import sys


def _load_state(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    sd = d.get("ema") or d.get("model") or d
    out = {}
    for k, v in sd.items():
        kk = k
        for pfx in ("module.", "_orig_mod."):
            if kk.startswith(pfx):
                kk = kk[len(pfx):]
        out[kk] = v
    return out, d


def _pick_keys(sd, needle):
    return sorted([k for k in sd if needle in k and sd[k].dim() >= 2])


def _row_cos_mean(a, b, eps=1e-8):
    """逐行余弦的平均（a,b 同形状 2D）。"""
    import torch
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return float((an * bn).sum(dim=-1).mean())


def _row_offdiag_cos(a, eps=1e-8):
    """行间余弦的**非对角**均值（越小越好；→1 表示所有行塌缩成一根）。"""
    import torch
    n = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    g = n @ n.t()
    m = g.shape[0]
    if m < 2:
        return float("nan")
    eye = torch.eye(m, dtype=torch.bool)
    return float(g[~eye].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--key", default="y_callig_embedder",
                    help="权重名里包含的子串（如 y_callig_embedder / E_pair）")
    ap.add_argument("--key2", default="", help="第二张表（可选，如 style_hier.E_pair）")
    ap.add_argument("--out", default="")
    ap.add_argument("--to", type=int, default=1000000, help="只看 <= 该步")
    a = ap.parse_args()

    import torch

    files = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")))
    files = [(int(re.search(r"(\d+)\.pt$", f).group(1)), f) for f in files]
    files = sorted([(s, f) for s, f in files if s <= a.to])
    if not files:
        print("没有 ckpt")
        sys.exit(1)

    sd0, meta0 = _load_state(files[0][1])
    keys = _pick_keys(sd0, a.key)
    keys2 = _pick_keys(sd0, a.key2) if a.key2 else []
    if not keys:
        print("找不到含 '%s' 的 2D 权重。可用候选：" % a.key)
        for k, v in sorted(sd0.items()):
            if v.dim() >= 2:
                print("   ", k, tuple(v.shape))
        sys.exit(1)

    print("=" * 78)
    print("T0 — 表运动轨迹  key='%s'  %d 个张量  ckpt 数=%d"
          % (a.key, len(keys), len(files)))
    print("=" * 78)
    print("ckpt dir: %s" % a.ckpt_dir)
    print("步范围:  %d -> %d" % (files[0][0], files[-1][0]))
    for k in keys:
        print("   %-52s %s" % (k, tuple(sd0[k].shape)))
    for k in keys2:
        print("   %-52s %s   [key2]" % (k, tuple(sd0[k].shape)))

    W0 = {k: sd0[k].float() for k in keys}
    W0b = {k: sd0[k].float() for k in keys2}

    rows = []
    print()
    print("%8s %11s %11s %11s %11s %11s" %
          ("step", "rel_move", "rowcos_ini", "row_cos_off", "row_norm_std", "rowcos_ini2"))
    print("-" * 78)
    for step, f in files:
        sd, _ = _load_state(f)
        rel, rc, rn, ro = [], [], [], []
        rc2 = []
        for k in keys:
            if k not in sd:
                continue
            W = sd[k].float()
            rel.append(float((W - W0[k]).norm() / W0[k].norm().clamp_min(1e-8)))
            if W.dim() >= 3:
                W = W.flatten(1)
                W0f = W0[k].flatten(1)
            else:
                W0f = W0[k]
            rc.append(_row_cos_mean(W, W0f))
            ro.append(_row_offdiag_cos(W))
            rn.append(float(W.norm(dim=-1).std()))
        for k in keys2:
            if k not in sd:
                continue
            W = sd[k].float()
            W0f = W0b[k] if W0b[k].dim() < 3 else W0b[k].flatten(1)
            W = W if W.dim() < 3 else W.flatten(1)
            rc2.append(_row_cos_mean(W, W0f))
        m = lambda x: sum(x) / len(x) if x else float("nan")
        r = {"step": step, "rel_move": m(rel), "row_cos_ini": m(rc),
             "row_cos_off": m(ro), "row_norm_std": m(rn),
             "row_cos_ini2": m(rc2) if rc2 else None}
        rows.append(r)
        print("%8d %11.5f %11.5f %11.5f %11.5f %11s"
              % (step, r["rel_move"], r["row_cos_ini"], r["row_cos_off"],
                 r["row_norm_std"],
                 ("%.5f" % r["row_cos_ini2"]) if r["row_cos_ini2"] is not None else "-"))

    # ---- 判读
    fin = rows[-1]
    print()
    print("[判读]")
    if fin["rel_move"] < 0.005:
        print("  ✗ 表几乎没动（rel_move=%.5f < 0.005）-> 梯度到不了表" % fin["rel_move"])
        print("     先查: --train-only-new-callig? weight decay? lr? 表是否被冻结?")
    elif fin["rel_move"] < 0.05:
        print("  ◐ 表动得很小（rel_move=%.5f）-> 有梯度但很弱" % fin["rel_move"])
    else:
        print("  ✓ 表明显在动（rel_move=%.5f）" % fin["rel_move"])

    if fin["row_cos_ini"] > 0.99:
        print("  表方向基本不变（row_cos_ini=%.5f）-> 只在原地缩放，没学新方向"
              % fin["row_cos_ini"])
    if fin["row_cos_off"] > 0.8:
        print("  ⚠ 行间余弦 %.3f 很高 -> 行在塌缩（不同书家的表行趋同）"
              % fin["row_cos_off"])
    else:
        print("  行间余弦 %.3f -> 行之间还分得开" % fin["row_cos_off"])

    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
