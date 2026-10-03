#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代码清理（只删确定安全的死代码，不动任何可能影响旧 ckpt 加载的东西）。

清理项:
  1. src/loss/style_rank_module.py 里的两处死代码:
     - `with torch.no_grad(): pass` 空块（无任何作用，纯残留）
     - `_enc_force_eval` 空方法（从未被调用）
"""
import sys

p = 'src/loss/style_rank_module.py'
s = open(p, encoding='utf-8').read()
n0 = len(s)

dead1 = """    @torch.no_grad()
    def _enc_force_eval(self):
        pass

"""
dead2 = """        with torch.no_grad():
            # 编码器全程冻结; 对 pred_xstart 不回传编码器参数, 但要回传 pred_xstart
            pass
"""
for d, tag in [(dead1, '_enc_force_eval 空方法'), (dead2, 'with no_grad: pass 空块')]:
    if d in s:
        s = s.replace(d, '')
        print(f'  ✓ 删除 {tag}')
    else:
        print(f'  - 未找到 {tag}（可能已清理）')

open(p, 'w', encoding='utf-8').write(s)
print(f'{p}: {n0} -> {len(s)} 字节')
