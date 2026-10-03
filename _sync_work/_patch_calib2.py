# -*- coding: utf-8 -*-
"""校准脚本 v2: (1) preload=True 让 skel latent 真的返回 (2) 打印 σ/k 全扫描曲线。"""
import io

P = "/root/Workspace/xy/DiT/tools/calibrate_mid_structure.py"
raw = io.open(P, encoding="utf-8", newline="").read()
src = "\n".join(raw.split("\r\n"))

REPS = [
    # 1) skel_lat 只在 preload=True 时填充 -> 非 preload 时 g=None, dilate 分支恒 inf
    ("        preload=False, load_image=False,",
     "        # \u2605 skel_lat \u53ea\u5728 preload=True \u65f6\u88ab\u586b\u5145 -> \u975e preload \u65f6 g=None, dilate \u5206\u652f\u6052 inf\n"
     "        preload=True, load_image=False,"),
    # 2) \u6253\u5370 \u03c3 \u5168\u626b\u63cf\u66f2\u7ebf
    ("            rb = [(resid(x0p, blur2d(x0, s)), s) for s in sgrid]\n"
     "            rb.sort()\n"
     "            res_b, sig = rb[0]",
     "            rb = [(resid(x0p, blur2d(x0, s)), s) for s in sgrid]\n"
     "            print(\"    \\u03c3-scan: \" + \"  \".join(f\"{s:g}:{r:.4f}\" for r, s in rb))\n"
     "            rb = sorted(rb)\n"
     "            res_b, sig = rb[0]"),
    # 3) \u6253\u5370 k \u5168\u626b\u63cf\u66f2\u7ebf
    ("                rd = [(resid(x0p, latent_dilate(g, k)), k) for k in kgrid]\n"
     "                rd.sort(); res_d, kk = rd[0]",
     "                rd = [(resid(x0p, latent_dilate(g, k)), k) for k in kgrid]\n"
     "                print(\"    k-scan: \" + \"  \".join(f\"{k:g}:{r:.4f}\" for r, k in rd))\n"
     "                rd = sorted(rd); res_d, kk = rd[0]"),
]

for old, new in REPS:
    assert old in src, "NOT FOUND: " + old[:60]
    src = src.replace(old, new)

out = "\r\n".join(src.split("\n"))
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("patched OK")
