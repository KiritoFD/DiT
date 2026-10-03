# -*- coding: utf-8 -*-
"""_chk_env.py — 检查远程各 python 环境的关键库可用性 (避免 ssh 内联引号问题)."""
import importlib
import sys

print('PY', sys.executable)
print('VER', sys.version.split()[0])
for m in ['numpy', 'scipy', 'PIL', 'skimage', 'skimage.morphology',
          'skimage.morphology.skeletonize', 'torch']:
    try:
        mod = importlib.import_module(m)
        print('OK  ', m, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print('MISS', m, type(e).__name__)