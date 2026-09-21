"""给 ocr_full_scan.py 加 GPU 开关。"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
p = "tools/ocr_full_scan.py"
s = io.open(p, encoding="utf-8").read()

old = """def _init():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR
    _OCR = RapidOCR()"""

new = """def _init():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR
    # ★ GPU: onnxruntime-gpu 1.16.3 + pip 的 nvidia-*-cu11 库
    #   （需配合 _sync_work/run_ocr_gpu.sh 设 LD_LIBRARY_PATH）
    #   实测 CPU 4.8/s -> GPU 39.4/s
    try:
        _OCR = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    except Exception:
        _OCR = RapidOCR()"""

if old in s:
    s = s.replace(old, new)
    io.open(p, "w", encoding="utf-8").write(s)
    print("  ✓ 加了 GPU 开关")
else:
    print("  ⚠ 没找到 _init（可能已改）")

# 单进程就够（GPU 下不需要多进程）
if '"--workers"' in s:
    s2 = io.open(p, encoding="utf-8").read()
    s2 = s2.replace('default=12)', 'default=1)')
    io.open(p, "w", encoding="utf-8").write(s2)
    print("  ✓ workers 默认改 1（GPU 单进程足够）")

import ast

ast.parse(io.open(p, encoding="utf-8").read())
print("  SYNTAX OK")
