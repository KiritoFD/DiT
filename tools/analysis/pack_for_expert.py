# -*- coding: utf-8 -*-
"""pack_for_expert.py — (1) 打出专家要的源文件包 (2) 写出 P0/P1 探针脚本

专家明确要的: 最新 v81 配置 + 修改后的 dit.py / Dataset / inference.py（此前 src.zip 没有这些）
并附: std_soft 编码器、trainer、eval、字体配置、store 的 index/provenance（不含图像, 只有元数据）
"""
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = "expert_pack/src_stdsoft_20261012"
os.makedirs(OUT, exist_ok=True)
FILES = [
    "src/model/dit.py",
    "src/utils/std_soft_glyph.py",
    "src/utils/latent_dataset.py",
    "src/eval/inference.py",
    "src/eval/in_mem_eval.py",
    "src/train/train.py",
    "src/train/cli.py",
    "src/train/configs/model_io.py",
    "src/loss/flow_matching.py",
    "configs/std_soft_fonts.json",
    "src/train/configs/v81_stdsoft_vs_v68_48.json",
    "src/train/configs/v81_stdsoft_vs_v68.json",
    "src/train/configs/v82_physics_noOT_stdsoft.json",
    "src/train/configs/v80_callivae_physics_noOT.json",
    "exp-std/data/std_soft_glyphs_v1/index.csv",
    "exp-std/data/std_soft_glyphs_v1/provenance.json",
    "tools/gen_eval_rows.py",
    "tools/build_std_soft_glyphs.py",
    "tools/analysis/white_baseline.py",
    "tools/analysis/v81_faintness.py",
    "tools/analysis/code_md5.py",
]
man = ["# std_soft / v81 塌陷问题 —— 给专家的源文件包\n",
       f"打包时间: 2026-10-12\n", "\n## 已确认的事实（供交叉验证）\n",
       "- v81（std_soft 空间条件）在 35k 墨量正常（0.174 vs GT 0.185），115k 大面积纯白（7/10 墨量 0.000）\n",
       "- 同一 ckpt 上：**已见字「惠」也白**；**重墨字「體」反而过墨**（0.4948 vs GT 0.3559）→ 非整体亮度漂移\n",
       "- 保存图与评测打印的 ssim 一致（实算 0.6371 vs 打印 0.6284）→ 取图没错\n",
       "- GT 与 CSV image_path 逐像素相同 → GT 映射没错\n",
       "- 白图零点：SSIM 0.6855 / SSIM-gauss 0.6494 / ink_iou 0.0000（后者对空白敏感 ✓）\n",
       "- index 8451 条 / 零缺字；store 平均墨量 0.0845\n", "\n## 本包文件\n"]
for f in FILES:
    if os.path.exists(f):
        d = os.path.join(OUT, f.replace("/", "__"))
        shutil.copyfile(f, d)
        man.append(f"- `{f}`  ({os.path.getsize(f)} B)\n")
    else:
        man.append(f"- (缺) `{f}`\n")
open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8", newline="").write("".join(man))
z = OUT + ".zip"
if os.path.exists(z):
    os.remove(z)
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(OUT):
        zf.write(os.path.join(OUT, f), f)
print(f"  -> {z}  ({os.path.getsize(z)/1e3:.0f} KB, {len(os.listdir(OUT))} 项)")

# ───────────────── P0/P1 探针（在 48 上跑） ─────────────────
PROBE = r'''#!/usr/bin/env python3
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
import csv, glob, os, sys
import numpy as np, torch
from PIL import Image
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
'''
os.makedirs("tools/analysis", exist_ok=True)
open("tools/analysis/probe_glyph_branch.py", "w", encoding="utf-8", newline="").write(PROBE)
print("  -> tools/analysis/probe_glyph_branch.py  (P0/P1 探针, 待按实际 API 微调后跑)")
