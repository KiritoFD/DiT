#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证: 模型能否加载离线形变头 + 风格是否真的驱动了 g'。

关键检查:
  ① deform_ckpt 载入无 missing/unexpected
  ② 模型的 _e_callig() 与离线训练用的表**同源**（否则头失效）
  ③ 换书家条件 -> g' 真的变（且幅度合理）
  ④ step0 不再恒等（因为载入了训好的权重）—— 这是**预期行为**
"""
import sys, json, os
import torch
sys.path.insert(0, '/root/Workspace/xy/DiT')
os.chdir('/root/Workspace/xy/DiT')
from src.train.cli import parse_args
from src.eval.model_io import build_model_from_args, apply_post_construction

a = parse_args(['--config', 'src/train/configs/v20_deform_skel_100k.json'])
# ⚠ 预训练书家表是在 apply_post_construction 里载入的; 只调 build_model_from_args
#   会得到一个**没载入表**的模型 -> 误报"风格源不同源"。
m = build_model_from_args(a, 'cpu')
apply_post_construction(m, a, verbose=False)
m.eval()
d = m.deform_skel
print()
print('① 形变头存在:', d is not None, '| 参数',
      format(sum(p.numel() for p in d.parameters()), ','))
print('   residual 开启:', d.res is not None, '| res_cap', d.res_cap,
      '| max_off', d.max_off)

# ② 风格源一致性: 模型的 _e_callig 等价物 vs 离线用的表
tab = torch.load('assets/callig_emb_pretrained_50k.pt', map_location='cpu',
                 weights_only=False)['embedding'].float()
with torch.no_grad():
    y = torch.arange(45)
    e_model = m.y_callig_embedder(y, False).float()
diff = float((e_model - tab).abs().max())
print(f'② 模型 y_callig_embedder vs 预训练表: max|Δ| = {diff:.6f} '
      f'{"✓ 同源" if diff < 1e-4 else "✗ 不同源 -> 头会失效!"}')

# ③ 换书家 -> g' 变化
g = torch.randn(1, 4, 32, 32)
with torch.no_grad():
    e0 = m.y_callig_embedder(torch.tensor([0]), False).float()
    e1 = m.y_callig_embedder(torch.tensor([7]), False).float()
    g_a = d(g, e0)
    g_b = d(g, e1)
print(f'③ 同 g、换书家(0->7): |g_a-g_b| mean = {float((g_a-g_b).abs().mean()):.4f} (>0 说明风格在驱动)')
print(f'   offset stats: {d.offset_stats()}')

# ④ 载入权重后 step0 不再恒等（预期）
print(f'④ |g\'-g| mean = {float((g_a-g).abs().mean()):.4f}  <- 载入训好权重后**应 >0**(不再是恒等)')
print()
print('判读: ①②③ 全过 = 链路通; ④>0 是预期(离线头已训好, 不是零初始化状态)')
