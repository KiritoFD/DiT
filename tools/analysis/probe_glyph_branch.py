#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_glyph_branch.py — P0/P1 探针（专家指定，不重训）

P1: glyph 分支的尺度与"背景 vs 内容"响应
     · 逐个 ckpt 打印所有含 glyph/std_soft 的模块树与参数
     · f0 = glyph_embedder(zeros)  ; fg = glyph_embedder(real g) ; 打印 ||f0||, ||fg||, ||fg-f0||
     · 若 ||f0|| >> ||fg-f0|| -> 条件编码器主要输出"空白画布特征"（专家假说 H2）
P0: 三种条件推理（normal / g=0 / wrong-char g）的墨量与 ink_iou
     正常 g 白、g=0 也白 -> 主干速度场已退化
     正常 g 白、g=0 恢复   -> 骨架注入支路可疑
"""
import glob, os, sys
import torch
os.chdir("/home/ds/Workspace/DiT"); sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CK = "assets/results/v81_stdsoft_vs_v68_48/20261011-225005-v81-stdsoft-vs-v68-48/checkpoints"
CFG = "src/train/configs/v81_stdsoft_vs_v68_48.json"
IDX = [131, 100, 67, 30, 177, 173, 86, 83, 145, 22]

import json
cfg = json.load(open(CFG, encoding="utf-8"))
from src.eval.model_io import build_model_from_args, load_model_from_ckpt   # 按实际 API 调整
class A: pass
a = A(); [setattr(a, k, v) for k, v in cfg.items()]

dev = torch.device("cuda:0")
for f in sorted(glob.glob(CK + "/*.pt")):
    print(f"\n=== {os.path.basename(f)} ===")
    try:
        model = build_model_from_args(a, dev)
        load_model_from_ckpt(model, f, device=dev)
        model.eval()
    except Exception as e:
        print("  构建/加载失败:", type(e).__name__, e); continue
    # --- P1: 找 glyph 相关模块 + 尺度 ---
    names = [n for n, _ in model.named_modules()
             if any(t in n.lower() for t in ("glyph", "soft", "skel"))]
    print("  相关模块:", names[:8])
    for nm in ("glyph_scale",):
        if hasattr(model, nm):
            print(f"  {nm} = {float(getattr(model, nm)):+.6f}")
    emb = None
    for n, m in model.named_modules():
        if "glyph" in n.lower() and "embed" in n.lower():
            emb = m; print("  embedder:", n, type(m).__name__); break
    if emb is not None:
        with torch.no_grad():
            g0 = torch.zeros(4, 1, 64, 64, device=dev)
            gr = torch.rand(4, 1, 64, 64, device=dev).round()      # 占位真 g（形状对即可）
            f0 = emb(g0); fg = emb(gr)
            print(f"  ||f0||={float(f0.norm()):.4f}  ||fg||={float(fg.norm()):.4f}  "
                  f"||fg-f0||={float((fg-f0).norm()):.4f}  "
                  f"内容/背景比={float((fg-f0).norm()/max(float(f0.norm()),1e-9)):.4f}")
        print("  ⚠ 上面用的是随机占位 g；真 g 需从 store 取（见 std_soft_glyph.StdSoftGlyphStore）")
print("\n  提示: f0 远大于 (fg-f0) => 专家假说 H2（背景/偏置响应淹没内容）成立风险高")
