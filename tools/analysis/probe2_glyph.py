#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe2_glyph.py — H2 判据：条件编码器的"背景/偏置响应" vs "真正内容响应"，跨 v81 ckpt

用法（在 48 上）: PYTHONPATH=. python /tmp/probe2_glyph.py
步骤:
  0) 打印 build_model_from_args / load_model_from_ckpt 的真实签名（先看再用）
  1) 逐个 ckpt 建模型+加载
  2) 列出所有含 glyph/soft/skel 的模块与可学习 scale
  3) 取真 std_soft 掩码（store）→ f_real ; 再取全零 → f0
     报告 ||f0||, ||f_real||, ||f_real-f0||, 内容/背景比
"""
import glob
import inspect
import json
import os
import sys

import torch

os.chdir("/home/ds/Workspace/DiT")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CK = "assets/results/v81_stdsoft_vs_v68_48/20261011-225005-v81-stdsoft-vs-v68-48/checkpoints"
CFG = "src/train/configs/v81_stdsoft_vs_v68_48.json"
STORE = "exp-std/data/std_soft_glyphs_v1"
IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]
dev = torch.device("cuda:0")

print("=== 0. 真实签名 ===")
try:
    from src.eval import model_io
    print("  build_model_from_args", inspect.signature(model_io.build_model_from_args))
    print("  load_model_from_ckpt ", inspect.signature(model_io.load_model_from_ckpt))
except Exception as e:
    print("  导入失败:", type(e).__name__, e)
    sys.exit(1)

cfg = json.load(open(CFG, encoding="utf-8"))


class A:
    pass


a = A()
for k, v in cfg.items():
    setattr(a, k, v)

# 真掩码
g_real = None
try:
    from src.utils.std_soft_glyph import StdSoftGlyphStore
    import csv
    rows = list(csv.DictReader(open(os.path.join(STORE, "index.csv"), encoding="utf-8-sig")))
    idx_map = {(int(r["script_id"]), r["character"]): r for r in rows}
    import csv as _c
    ev = list(_c.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))
    tens = []
    for i in IDX:
        sid = int(ev[i].get("script_id") or 0)
        ch = ev[i]["character"]
        st = StdSoftGlyphStore(STORE, is_train=False, preload=False)
        tens.append(st.get(sid, ch))
    g_real = torch.stack(tens).to(dev)
    print(f"  真掩码: {tuple(g_real.shape)}  墨量均值={float(g_real.mean()):.4f}")
except Exception as e:
    print("  取真掩码失败:", type(e).__name__, e)

print("\n=== 1-3. 逐 ckpt ===")
print(f"  {'ckpt':>10} | {'glyph 模块':>10} | {'scale':>10} | {'||f0||':>9} | {'||f_real||':>10} | "
      f"{'||f_real-f0||':>12} | {'内容/背景':>9}")
for f in sorted(glob.glob(CK + "/*.pt")):
    tag = os.path.basename(f).replace(".pt", "")
    try:
        model = model_io.build_model_from_args(a, dev)
        _out = model_io.load_model_from_ckpt(f, device=dev, use_ema=True, verbose=False)
        if isinstance(_out, tuple):
            model = _out[0]
        elif _out is not None:
            model = _out
        model.eval()
    except Exception as e:
        print(f"  {tag:>10} | 构建/加载失败: {type(e).__name__}: {str(e)[:70]}")
        continue
    mods = [(n, m) for n, m in model.named_modules()
            if any(t in n.lower() for t in ("glyph", "soft", "skel"))]
    scales = {n: float(p.detach()) for n, p in model.named_parameters()
              if p.numel() == 1 and any(t in n.lower() for t in ("glyph", "soft", "skel"))}
    emb = None
    for n, m in mods:
        if "embed" in n.lower():
            emb = (n, m)
            break
    if emb is None:
        print(f"  {tag:>10} | 模块 {len(mods)} 个, 但没找到 embedder; 名字示例 {[n for n,_ in mods[:4]]}")
        continue
    nm, m = emb
    try:
        with torch.no_grad():
            f0 = m(torch.zeros(4, 1, 64, 64, device=dev))
            fr = m(g_real[:4])
            n0, nr = float(f0.norm()), float((fr - f0).norm())
        print(f"  {tag:>10} | {len(mods):>10} | {str(list(scales.values())[:1]):>10} | {n0:>9.4f} | "
              f"{float(fr.norm()):>10.4f} | {nr:>12.4f} | {nr/max(n0,1e-9):>9.4f}")
    except Exception as e:
        print(f"  {tag:>10} | embedder 前向失败: {type(e).__name__}: {str(e)[:60]}  (模块={nm})")
print("""
  判读（专家 H2）:
   * 内容/背景比 = ||f_real-f0|| / ||f0||
     若它 **随训练单调下降** -> 条件编码器后期主要输出"背景/偏置响应"，真字符结构被淹没 -> H2 成立
     若它平稳 -> H2 被否，转查 H1（注入范数/scale 漂移）或主干速度场(P2)
""")
