"""Small causal checks for geometry, Pair sign/budget and nested protocols."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from src.unet.model import LatentUNet
from src.unet.velocity import WriterVelocityUNet
from tools.style_matrix.velocity_pair import backward_pair,PairBudget
from tools.style_matrix.runtime.common import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(8);device='cuda'
    g=torch.ones(4,1,32,32,device=device);g[:,:,8:24,10:22]=0
    geom=WriterVelocityUNet.geometry(g)
    truth=distance_transform_edt((g[0,0].cpu().numpy()<.5))
    assert np.allclose(geom[0,3].cpu().numpy(),np.minimum(truth,16)/8,atol=1e-6)
    x=torch.randn(4,4,32,32,device=device)*.2;g=torch.nn.functional.interpolate(g,size=(64,64))
    w=torch.tensor([0,1,0,1],device=device);s=torch.zeros(4,dtype=torch.long,device=device);t=x.new_full((4,),1000.)
    results={}
    for mode in ('cnn','geometry','attention'):
        model=WriterVelocityUNet(LatentUNet((32,64,64),num_res_blocks=1,num_writers=2,num_scripts=1),mode,width=32).cuda().eval()
        with torch.no_grad():
            model.output[-1].weight.normal_(std=.01)
            # Residual convs start at zero, so test a reachable trained state.
            model.blocks[0][0].conv2.weight.normal_(std=.01)
            content,style=model.components(x,t,w,g,s,True)
            other=model.components(x,t,1-w,g,s,True)
            null=model.components(x,t,torch.full_like(w,2),g,s,True)
            assert torch.equal(content,other[0]) and torch.equal(content,null[0])
            assert torch.count_nonzero(null[1])==0
            assert float((style-other[1]).abs().max())>0
        model.train();target=torch.randn_like(x)
        loss=(model(x,t,w,g=g,y_script=s,condition_dropped=True)-target).square().mean();loss.backward()
        rows={i:dict(character='永',script='楷',calligrapher=str(i%2)) for i in range(4)}
        budget=PairBudget();budget.backward(model,x,g,w,s,dict(img_id=torch.arange(4)),rows,1.,4,0)
        assert abs(budget.state['ratio']-.25)<1e-5,budget.state
        restored=PairBudget(copy.deepcopy(budget.state));assert restored.state==budget.state
        results[mode]=dict(null_exact=True,content_writer_independent=True,pair_budget=budget.state)
        del model
    class Oracle(torch.nn.Module):
        def __init__(self):super().__init__();self.values=torch.nn.Parameter(torch.tensor([-.1,-.3],device=device))
        def forward(self,x,t,w,**kw):return self.values[w,None,None,None].expand_as(x)
    z=torch.stack([torch.full_like(x[0],.1),torch.full_like(x[0],.3)])
    value=backward_pair(Oracle(),z,g[:2],w[:2],s[:2],dict(img_id=torch.arange(2)),rows,1.,1.,2)
    assert float(value)<1e-10,value
    save(a.out,dict(results=results,pair_target_sign_correct=True,edt_matches_scipy=True))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
