# -*- coding: utf-8 -*-
"""make_v83.py — 生成 v83-A / v83-B 配置（v81 配方 + 零输入响应消除；B 另加 glyph_drop_prob=0.1）

v81 = v68 逐项照抄 + std_soft ⇒ **v83-A vs v81 是"修复是否有效"的干净单变量对照** ✓
（同 SD-VAE shards / 同 aug 语料 / 同 batch 288 / 同 200k cosine / 同 C2OT ✓）
v80 只作"稳定基线"参照（与 v83 相差 VAE/语料/OT/SDE 四个变量 ✗，不能用来归因）
"""
import json
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_R if os.path.isdir(os.path.join(_R, "src")) else os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

src = "src/train/configs/v81_stdsoft_vs_v68_48.json"
d0 = json.load(open(src, encoding="utf-8"))

for tag, drop in (("A", 0.0), ("B", 0.1)):
    d = dict(d0)
    d["experiment_name"] = f"v83{tag}-stdsoft-zerofix"
    d["results_dir"] = f"assets/results/v83{tag}_stdsoft_zerofix"
    d["glyph_drop_prob"] = float(drop)
    d["_comment"] = (
        f"v83-{tag} = v81 配方 + **零输入响应消除**（dit.py: feat = enc(g) - enc(0)，不 detach）"
        + (f" + **glyph_drop_prob={drop}**（训练中真正见过'无条件'）" if drop else "")
        + "\n\n根因(2026-10-12 实测): ‖encoder(0)‖ 20k→115k 由 19.93 涨到 72.07(3.6×)，"
          "内容/背景比 2.48→1.13，墨量 0.174→0.073 ⇒ 输出塌成平均画布(白纸)。"
          "内因: 编码器前两层 Conv2d 带 bias + glyph_drop_prob 默认 0（从未训练空条件）。\n"
          "对照纪律: **v83-A vs v81 才是单变量因果检验**（同 VAE/语料/batch/调度/C2OT）；"
          "v80 只作稳定基线参照（与 v83 相差 4 个变量）。\n"
          "硬闸门: 软墨量 < GT 的 10% 记近空白；连续两个 ckpt 墨量明显下滑即停，"
          "不再因 SSIM 上升而继续。验收点 20k/35k/50k/70k，ODE γ=0，固定样本+seed，"
          "并单列 112 个未见组合。")
    out = f"src/train/configs/v83{tag}_stdsoft_zerofix.json"
    json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {out}   glyph_drop_prob={drop}")
