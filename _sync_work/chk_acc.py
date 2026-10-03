import sys

print("  python:", sys.executable)
try:
    import accelerate
    print("  accelerate:", accelerate.__version__, accelerate.__file__)
except Exception as e:
    print("  ✗ accelerate import 失败:", type(e).__name__, str(e)[:120])

try:
    import transformers
    print("  transformers:", transformers.__version__)
except Exception as e:
    print("  ✗ transformers:", e)

try:
    from transformers.integrations.accelerate import is_accelerate_available
    print("  is_accelerate_available():", is_accelerate_available())
except Exception as e:
    print("  ✗ 检查函数:", type(e).__name__, str(e)[:120])
