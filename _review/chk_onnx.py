import os
import sys

print("  python:", sys.executable)
try:
    import onnxruntime as ort
    print("  onnxruntime:", ort.__version__)
    print("  providers:", ort.get_available_providers())
    # 实际测一次 CUDA provider 能否初始化
    try:
        import numpy as np
        so = ort.SessionOptions()
        sess = ort.InferenceSession(
            None, so, providers=["CUDAExecutionProvider"])
        print("  CUDA EP 初始化: ✓")
    except Exception as e:
        print("  CUDA EP 初始化: ✗", type(e).__name__, str(e)[:120])
except Exception as e:
    print("  ✗ onnxruntime:", type(e).__name__, str(e)[:120])

print("\n  === 已缓存的 OCR 模型 ===")
for d in ("/root/.paddleocr", "/root/.rapidocr", "/root/.cache/paddleocr",
          "/root/.EasyOCR", "/root/.cache/modelscope"):
    if os.path.isdir(d):
        print(f"  {d}:")
        for root, dirs, files in os.walk(d):
            lv = root[len(d):].count(os.sep)
            if lv > 3:
                continue
            for f in files[:6]:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p) / 1e6
                    print(f"    {p}  {sz:.1f}MB")
                except OSError:
                    pass
