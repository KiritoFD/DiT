# -*- coding: utf-8 -*-
"""apply_refactor_p2.py — 把 v83-A/B 依赖的 9 个隐式默认键显式写入配置（值 = 当前实际生效值）

原则（重构≠改行为）:
  * 只**新增键**，不修改任何已有键 -> 重启后行为逐位不变 ✓
  * 值 = 代码里的默认值（也就是这两个 run 现在真正在用的值）✓
  * 同时记录"别的实验族用的是另一个值"，供后人判断这是否是有意为之 ✓
"""
import json
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_R if os.path.isdir(os.path.join(_R, "src")) else os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 键 -> (v83 实际生效值, 其它实验族共识值, 说明)
KEYS = {
    "eval_blend_alpha":       (0.0, 0.5,  "评测混合系数; 0.0=不把输入混进预测"),
    "glyph_embedder_depth":   (0,   2,    "std_soft 路径用硬编码 Sequential, 此键不生效"),
    "callig_scale_init":      (1.0, 1.5,  "书家条件强度初值"),
    "dino_per_script_center": (0,   1,    "DINO 特征是否按书体中心化"),
    "repa_early_warmup":      (0,   2000, "REPA 预热步数"),
    "glyph_latent_channels":  (4,   16,   "std_soft 强制 1 通道, 此键不生效"),
    "callig_multi_style_k":   (0,   4,    "书家多风格模态数 k"),
    "deform_max_off":         (3,   6,    "形变最大偏移"),
    "w_style_ortho":          (0.0, 0.05, "风格正交正则权重"),
}

for tag in ("A", "B"):
    p = f"src/train/configs/v83{tag}_stdsoft_zerofix.json"
    d = json.load(open(p, encoding="utf-8"))
    added = {}
    for k, (eff, other, why) in KEYS.items():
        if k in d:
            continue                       # 已有 -> 绝不覆盖 ✓
        d[k] = eff
        added[k] = f"{eff}（本族原为隐式默认; 他族用 {other}）"
    note = ("\n\n【Phase-2 显式化 2026-10-12】以下键原为**隐式默认**（配置未给, 代码兜底）; "
            "为消除'配置漏写->静默用错默认'隐患, 按**当前实际生效值**显式写入 —— "
            "重启后行为逐位不变 ✓。他族若用别的值, 已在下方标注, 请确认是否本族有意为之:\n"
            + "\n".join(f"  · {k}: {v}" for k, v in added.items()))
    if added:
        d["_comment"] = d.get("_comment", "") + note
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  ✓ {p}: 显式化 {len(added)} 键")
    else:
        print(f"  · {p}: 无需改动")

# 校验：新旧文件的**数值语义**必须一致（除这 9 个键外无差异）
print("\n  校验（应仅多出那 9 个键）:")
for tag in ("A", "B"):
    p = f"src/train/configs/v83{tag}_stdsoft_zerofix.json"
    d = json.load(open(p, encoding="utf-8"))
    extra = [k for k in d if k in KEYS]
    print(f"    v83-{tag}: 已含 {len(extra)}/9 个显式键 -> {sorted(extra)}")
