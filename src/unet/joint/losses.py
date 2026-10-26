"""Continuous image losses used by the optional decoder objective."""
import torch
from torch.nn import functional as F


def edge_loss(pred,target):
    k=pred.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    kernels=torch.stack((k,k.t()))[:,None]
    total=pred.new_zeros(())
    for scale in (1,2,4):
        p,t=pred.mean(1,keepdim=True),target.mean(1,keepdim=True)
        if scale>1:p,t=F.avg_pool2d(p,scale),F.avg_pool2d(t,scale)
        total=total+F.l1_loss(F.conv2d(p,kernels,padding=1),F.conv2d(t,kernels,padding=1))/3
    return total
