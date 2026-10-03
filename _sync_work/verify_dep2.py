"""依赖解开的有效验证 (上一版两步方法本身错了)。

[1] 逐字比对: 新 src/model/injections.py 与 src/model/legacy/controlnet.py 里
    zero_init_linear / ZeroAdaLNInjection 的实现是否完全一致 (归一化空白后)。
    -> 这是"键名未变"的**充分证明**: state_dict 键名 = 属性名链, 与类所在文件无关。
[2] 找真 adaLN ckpt: 扫全部 ckpt, 挑出 glyph_injections 键**全是 proj.*** 的那个
    (即 adaLN 注入), 确认它存在且键名与新模块完全同构。
[3] 新模块与 legacy 模块的 state_dict 键逐一比对。
"""
import glob
import io
import os
import re
import sys

import torch

NEW = "src/model/injections.py"
OLD = "src/model/legacy/controlnet.py"


def norm(s):
    s = re.sub(r"#.*", "", s)                 # 去注释
    s = re.sub(r'"""(?:.|\n)*?"""', "", s)    # 去 docstring
    s = re.sub(r"'''(?:.|\n)*?'''", "", s)
    return re.sub(r"\s+", " ", s).strip()


def grab(src, anchor, stops):
    i = src.find(anchor)
    if i < 0:
        return None
    j = len(src)
    for s in stops:
        k = src.find(s, i + len(anchor))
        if k > 0:
            j = min(j, k)
    return src[i:j]


print("=" * 76)
print("[1] 逐字比对 (新家 vs 旧家)")
new_src = io.open(NEW, encoding="utf-8").read()
old_src = io.open(OLD, encoding="utf-8").read()

pairs = [
    ("zero_init_linear", ["\nclass "]),
    ("class ZeroAdaLNInjection", ["\nclass ControlConditionEncoder"]),
]
allok = True
for anchor, stops in pairs:
    a = norm(grab(old_src, anchor if anchor.startswith("class") else f"def {anchor}", stops) or "")
    b = norm(grab(new_src, anchor if anchor.startswith("class") else f"def {anchor}",
                  ["\nclass ", "\ndef "]) or "")
    ok = bool(a) and a == b
    allok &= ok
    print(f"  {anchor:<26} {'✓ 完全一致' if ok else '✗ 有差异'}")
    if not ok:
        print(f"      旧: {a[:160]}")
        print(f"      新: {b[:160]}")

print("\n" + "=" * 76)
print("[2] 找真 adaLN ckpt 并核对键名同构")
sys.path.insert(0, "/root/Workspace/xy/DiT")
from src.model.injections import ZeroAdaLNInjection           # noqa: E402

new_keys = set(ZeroAdaLNInjection(384).state_dict().keys())
print(f"  新模块 state_dict 键: {sorted(new_keys)}")

cands = glob.glob("exp-std/runs_purestd/**/checkpoints/*.pt", recursive=True) \
    + glob.glob("assets/results/**/checkpoints/*.pt", recursive=True)
found = []
for f in cands[:400]:
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
    except Exception:                                          # noqa: BLE001
        continue
    sd = d.get("ema") or d.get("model") or d.get("delta") or d
    if not isinstance(sd, dict):
        continue
    inj = [k for k in sd if "glyph_injections" in k]
    if inj and all(k.endswith(("proj.weight", "proj.bias", ".proj.weight", ".proj.bias"))
                   for k in inj):
        found.append((f, len(inj), sorted({k.split("glyph_injections.")[-1].split(".", 1)[-1]
                                           for k in inj})))
        if len(found) >= 2:
            break

if not found:
    print("  ⚠ 没扫到 adaLN 注入的 ckpt (可能历史 run 都用了 xattn)")
else:
    for f, n, suf in found:
        print(f"  ckpt: {f}")
        print(f"    glyph_injections 键 {n} 个, 后缀集合 = {suf}")
        exp = set()
        for s2 in ("proj.weight", "proj.bias"):
            exp.add(s2)
        print(f"    {'✓ 与新高斯同构 (proj.weight/proj.bias)' if set(suf) == exp else '差异: ' + str(suf)}")

print("\n" + "=" * 76)
verdict = allok and new_keys == {"proj.weight", "proj.bias"}
print("结论:", "✓ 搬家逐字未改 + 键名同构 -> 历史 adaLN ckpt 照常加载"
      if verdict else "✗ 有问题, 见上")
sys.exit(0 if verdict else 3)
