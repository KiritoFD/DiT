#!/usr/bin/env python3
"""兼容 wrapper: tools/eval/eval_stdskel_batch.py -> src.eval.batch_eval (轮子已迁移)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.eval.batch_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
