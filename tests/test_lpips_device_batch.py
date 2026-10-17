import numpy as np
import pytest
import torch
from src.eval import decoder


def test_lpips_microbatch_preserves_order_and_values(monkeypatch):
    seen=[]
    class Distance(torch.nn.Module):
        def __init__(self):super().__init__();self.marker=torch.nn.Parameter(torch.tensor(0.))
        def forward(self,p,g):
            assert p.device==g.device==self.marker.device
            assert len(p)<=8
            seen.append(len(p))
            return (p-g).square().flatten(1).mean(1)
    monkeypatch.setattr(decoder,'_get_lpips',lambda:Distance())
    p=np.arange(17,dtype=np.float32)[:,None,None,None]/16
    p=np.broadcast_to(p,(17,4,4,3)).copy();g=np.zeros_like(p)
    result=decoder._lpips_per_sample(p,g)
    assert seen==[8,8,1]
    assert result==pytest.approx(4*(np.arange(17)/16)**2)
