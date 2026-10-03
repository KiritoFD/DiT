#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用规范入口构造模型，验证 2 层 GlyphQuery 接对了。"""
import json
import sys

sys.path.insert(0, '.')
from src.train.cli import parse_args  # noqa: E402
from src.eval.model_io import build_model_from_args  # noqa: E402

d = json.load(open('src/train/configs/v17_inj3_fixed_aug_100k.json'))
d['local_ca_layers'] = 2
d['local_ca_at'] = '2,6'
json.dump(d, open('/tmp/_gq.json', 'w'))
a = parse_args(['--config', '/tmp/_gq.json'])
print('local_ca_layers =', a.local_ca_layers, '| local_ca_at =', a.local_ca_at)

m = build_model_from_args(a, 'cpu')
print('local_ca 层数:', len(m.local_ca), '| 类型:', type(m.local_ca[0]).__name__)
print('_local_ca_map:', m._local_ca_map)
print('out_log_scale exp 初值:', [round(float(lc.out_log_scale.exp()), 4) for lc in m.local_ca])
print('out_proj zero-init:', [bool((lc.out_proj.weight == 0).all()) for lc in m.local_ca])
print('style_to_q zero-init:', [bool((lc.style_to_q.weight == 0).all()) for lc in m.local_ca])
print('q_proj std:',
      [round(float(lc.q_proj.weight.std()), 5) for lc in m.local_ca])
print('win_mask:', [None if lc.win_mask is None else tuple(lc.win_mask.shape) for lc in m.local_ca])
print('总参数:', f'{sum(p.numel() for p in m.parameters()):,}')
print('GlyphQuery 参数:', f'{sum(p.numel() for lc in m.local_ca for p in lc.parameters()):,}')
