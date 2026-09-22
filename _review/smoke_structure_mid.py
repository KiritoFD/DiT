# -*- coding: utf-8 -*-
import ast, sys, torch
files = ["src/loss/structure_mid.py", "tools/build_std_skel_widths.py",
         "tools/calibrate_mid_structure.py", "src/train/train.py", "src/train/cli.py"]
for f in files:
    ast.parse(open(f, encoding="utf-8").read())
print("AST OK:", files)

sys.path.insert(0, ".")
from src.loss.structure_mid import (MidStructureLoss, TSchedule, blur2d,
                                    latent_dilate, lowpass, chan_norm)
torch.manual_seed(0)
N, C, H = 4, 4, 32
pred = torch.randn(N, C, H, H, requires_grad=True)
gt = torch.randn(N, C, H, H)
skel = torch.randn(N, C, H, H)
t = torch.tensor([0.2, 0.45, 0.55, 0.9])   # 混合: 有的落窗口[0.25,0.65](a_t=1-t)内

for carrier in ("blur_gt", "dilate_skel", "both"):
    loss = MidStructureLoss(carrier=carrier, latent_channels=C)
    val = loss(pred, gt, skel, t)
    val.backward()
    print(f"carrier={carrier:12s} loss={val.item():.5f} grad_ok={pred.grad is not None}")
    pred.grad = None

# 带 calib schedule
sch = TSchedule(grid_t=[0.3,0.5,0.7], grid_sigma=[0.5,1.5,3.0], grid_k=[1,2,3])
loss = MidStructureLoss(carrier="blur_gt", sigma_sched=sch, lp_factor=2)
print("calib σ(0.5)=", float(sch.sigma_of(torch.tensor([0.5]))), " k(0.5)=", float(sch.k_of(torch.tensor([0.5]))))
print("calib loss=", float(loss(pred, gt, skel, t)))
# 窗口外应返回 0
t_out = torch.tensor([0.05, 0.05, 0.05, 0.05])   # a_t=0.95 > ahi -> 不 active
print("out-of-window loss (应=0):", float(MidStructureLoss()(pred, gt, skel, t_out)))
print("ALL SMOKE OK")
