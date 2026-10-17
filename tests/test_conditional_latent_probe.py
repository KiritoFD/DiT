import numpy as np
import torch
from tools.analysis.conditional_latent_probe import fit_projection


def test_conditional_projection_prefers_writer_over_content():
    torch.manual_seed(89)
    rows=[];blocks=[]
    for _ in range(100):
        c=torch.randn(2)*4
        start=len(rows)
        for w,ci in ((1,0),(1,1),(-1,0),(-1,1)):
            rows.append([float(w),float(c[ci]),float(torch.randn(())*.01)])
        blocks.append(list(range(start,start+4)))
    x=torch.tensor(rows)
    mean,basis,w,values=fit_projection(x,torch.tensor(blocks),rank=3,output=1)
    direction=(basis@w).flatten()
    assert abs(direction[0])>10*abs(direction[1])
    assert torch.isfinite(values).all()
    assert torch.allclose((x-mean)@basis@w,(x-mean)@(basis@w),atol=1e-5)


def test_fit_is_independent_of_heldout_values():
    torch.manual_seed(89)
    train=torch.randn(80,12);blocks=torch.arange(80).reshape(-1,4)
    torch.manual_seed(89);a=fit_projection(train,blocks,rank=8,output=3)
    heldout=np.full((20,12),1e6)  # deliberately not supplied to fit
    assert heldout.shape==(20,12)
    torch.manual_seed(89);b=fit_projection(train,blocks,rank=8,output=3)
    for aa,bb in zip(a,b):assert torch.equal(aa,bb)
