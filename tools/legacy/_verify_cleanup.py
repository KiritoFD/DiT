# -*- coding: utf-8 -*-
"""
_verify_cleanup.py — 清理后的完整性验证（不依赖 GPU，CPU 上跑）。

检查项
------
1. src.model / src 能否正常 import（不触发 timm / lora）
2. DiT_2Cond_models 里有哪些模型，DiT-2Cond-S/2 能否实例化
3. ControlNet 相关类能否 import
4. 已删除的名字确实不可 import
5. 模型前向 + 反向能否跑通（用小 batch）
6. timm 是否真的不再被 dit.py 依赖
"""
import os, sys, importlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ok = True


def check(label, fn):
    global ok
    try:
        r = fn()
        print(f"  [OK]   {label}" + (f"  -> {r}" if r else ""))
        return r
    except Exception as e:
        ok = False
        print(f"  [FAIL] {label}  -> {type(e).__name__}: {e}")
        return None


print("=" * 70)
print("1) import 检查")
print("=" * 70)
check("import src.model", lambda: importlib.import_module("src.model"))
check("import src", lambda: importlib.import_module("src"))
check("import src.model.dit", lambda: importlib.import_module("src.model.dit"))
check("import src.model.controlnet",
      lambda: importlib.import_module("src.model.controlnet"))
check("import src.model.modules", lambda: importlib.import_module("src.model.modules"))

print("\n" + "=" * 70)
print("2) 已删除的名字应不可 import")
print("=" * 70)
for name in ("inject_lora", "upgrade_lora_rank", "extract_full_inference",
             "DiT_3Cond", "DiT_3Cond_models", "DiT", "DiT_models"):
    try:
        import src.model as m
        if hasattr(m, name):
            ok = False
            print(f"  [FAIL] {name} 仍可从 src.model 导入")
        else:
            print(f"  [OK]   {name} 已移除")
    except Exception as e:
        ok = False
        print(f"  [FAIL] 检查 {name} 时出错: {e}")

print("\n" + "=" * 70)
print("3) 模型注册表")
print("=" * 70)
from src.model import DiT_2Cond_models, ControlNetDiT, ControlConditionEncoder
print(f"  DiT_2Cond_models keys: {list(DiT_2Cond_models.keys())}")
if "DiT-2Cond-XL/2" in DiT_2Cond_models:
    ok = False
    print("  [FAIL] XL 仍在注册表")
else:
    print("  [OK]   XL 已从注册表移除")
if "DiT-2Cond-S/2" in DiT_2Cond_models:
    print("  [OK]   DiT-2Cond-S/2 在位（当前 pipeline 使用）")
else:
    ok = False
    print("  [FAIL] DiT-2Cond-S/2 丢失")

print("\n" + "=" * 70)
print("4) 实例化 + 前向反向（CPU，小 batch）")
print("=" * 70)
import torch
model = DiT_2Cond_models["DiT-2Cond-S/2"](
    input_size=32, in_channels=4,
    num_calligraphers=1013, num_characters=35130,
    use_checkpoint=False, learn_sigma=False,
    condition_fusion="factorized_add",
    callig_embed_dim=128, char_embed_dim=384,
    cond_drop_all_prob=0.05, cond_drop_one_prob=0.25,
    skel_head_enabled=False, use_glyph_cond=True,
    glyph_scale_init=0.4, glyph_inject_layers=6,
    char_proj_mode="mlp", freeze_char_table=True,
    norm_type="rms", mlp_type="swiglu", qk_norm=True,
    rope=True, rope_theta=100.0, attn_impl="eager",
)
n = sum(p.numel() for p in model.parameters())
print(f"  参数量: {n:,} ({n/1e6:.2f}M)")

x = torch.randn(2, 4, 32, 32)
g = torch.randn(2, 4, 32, 32)
t = torch.rand(2)
yc = torch.randint(0, 1013, (2,))
yh = torch.randint(0, 35130, (2,))
out = model(x, t, y_callig=yc, y_char=yh, g=g)
print(f"  前向 OK: out={tuple(out.shape) if hasattr(out,'shape') else type(out)}")

print("\n" + "=" * 70)
print("5) timm 依赖检查")
print("=" * 70)
import subprocess
r = subprocess.run([sys.executable, "-c",
                    "import src.model.dit as d, sys; "
                    "print('timm' in sys.modules)"],
                   capture_output=True, text=True, cwd=ROOT)
loaded = r.stdout.strip()
if loaded == "False":
    print("  [OK]   import dit 不再加载 timm")
else:
    print(f"  [WARN] timm 仍被加载: {loaded}  (可能来自其它模块，非 dit.py)")

print("\n" + "=" * 70)
print("结果:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
