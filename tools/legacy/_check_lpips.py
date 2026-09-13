"""Check if lpips is installed on remote."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    import lpips
    print("lpips OK, version:", getattr(lpips, "__version__", "unknown"))
except ImportError as e:
    print(f"lpips NOT installed: {e}")

try:
    import torch
    print("torch OK:", torch.__version__)
except ImportError as e:
    print(f"torch NOT installed: {e}")
