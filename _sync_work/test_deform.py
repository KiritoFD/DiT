#!/usr/bin/env python3
import sys, torch, torch.nn as nn
sys.path.insert(0, '/root/Workspace/xy/DiT')
from src.model.deform_skel import DeformSkel
torch.manual_seed(0)
m = DeformSkel(cond_dim=128, ch=4, grid=32)
print('参数', format(sum(p.numel() for p in m.parameters()), ','))
g = torch.randn(2, 4, 32, 32); e = torch.randn(2, 128)
o = m(g, e)
print('① step0 |out-g| max =', float((o - g).abs().max()), '  <- 应为 0')
print('   offset:', m.offset_stats())
with torch.no_grad():
    nn.init.normal_(m.out.weight, std=0.02); nn.init.normal_(m.out.bias, std=0.02)
g2 = g.clone().requires_grad_(True); e2 = e.clone().requires_grad_(True)
o2 = m(g2, e2)
print('② 打散后 |out-g| max =', float((o2 - g).abs().max()), '  <- 应 >0')
o2.pow(2).mean().backward()
print('   d/d(g) norm =', '%.4f' % float(g2.grad.norm()),
      ' d/d(style) norm =', '%.4f' % float(e2.grad.norm()))
print('   offset:', m.offset_stats())
o3 = m(g, torch.randn(2, 128))
print('③ 换风格 |out_a-out_b| mean =', '%.5f' % float((o2 - o3).abs().mean()), '  <- 应 >0')
print('④ 形状/NaN:', tuple(o.shape), bool(torch.isnan(o).any()))
