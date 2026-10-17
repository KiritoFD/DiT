"""Finite scientific interface gates, including the CUDA hierarchical tail batch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.unet.io import Config,ARCHITECTURE,build,load,diffusion
from src.unet.blocks import ResBlock,timestep_embedding
from src.eval.sampler import sample_latents


class Echo(torch.nn.Module):
    std_soft_cond=True
    in_channels=4
    def forward(self,x,t,y_callig,y_char=None,g=None,y_script=None,y_callig_raw=None,y_pair=None):
        return torch.zeros_like(x)+(y_callig+y_script*10+y_callig_raw*100+y_pair*1000)[:,None,None,None]
    def forward_with_cfg(self,x,t,cfg_scale=1,**kwargs):return self(x,t,**kwargs)


class Once:
    def ddim_sample_loop(self,fn,shape,noise,**kwargs):
        return fn(noise,torch.zeros(len(noise),device=noise.device),**kwargs['model_kwargs'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');p.add_argument('--out')
    a=p.parse_args();torch.set_num_threads(4)
    cfg=Config(channels=(64,128,192));m=build(cfg).eval()
    for block in m.modules():
        if isinstance(block,ResBlock):torch.nn.init.normal_(block.conv2.weight,std=.001)
    torch.nn.init.normal_(m.output[-1].weight,std=.02)
    x=torch.randn(2,4,32,32);g=torch.rand(2,1,64,64);w=torch.tensor([0,1]);s=torch.tensor([0,1]);t=torch.tensor([200.,800.])
    y=m(x,t,w,g=g,y_script=s)
    assert y.shape==x.shape
    assert not torch.equal(y,m(x,t,w.flip(0),g=g,y_script=s))
    assert not torch.equal(y,m(x,t,w,g=1-g,y_script=s))
    assert not torch.equal(y,m(x,t,w,g=g,y_script=s.flip(0)))
    assert torch.equal(y,m.forward_with_cfg(x,t,w,g=g,y_script=s,cfg_scale=1))
    null=m(x,t,torch.full_like(w,cfg.num_writers),g=g,y_script=s)
    torch.testing.assert_close(m.forward_with_cfg(x,t,w,g=g,y_script=s,cfg_scale=2),null+2*(y-null))
    y.mean().backward()
    for value in (m.writer.weight,m.script.weight,m.std_stem[0].weight):assert value.grad.abs().sum()>0
    captured=[];hook=m.register_forward_pre_hook(lambda module,args:captured.append(args[1].clone()))
    diffusion(cfg).training_losses(m,x,torch.tensor([.2,.8]),{'y_callig':w,'y_script':s,'g':g})
    hook.remove();torch.testing.assert_close(captured[0],t)
    with tempfile.TemporaryDirectory() as temp:
        checkpoint=Path(temp)/'smoke.pt'
        torch.save({'architecture':ARCHITECTURE,'config':asdict(cfg),'ema':m.state_dict(),'train_steps':0},checkpoint)
        clone,_,_=load(checkpoint)
        torch.testing.assert_close(clone(x,t,w,g=g,y_script=s),y)
    records={'shape':True,'writer_script_std_sensitive':True,'cfg1_direct':True,
             'cfg_writer_null_keeps_std_script':True,'time_scale_1000':True,'strict_restore':True}
    if a.cuda:
        echo=Echo();noise=torch.zeros(5,4,32,32);conds=[(i,i) for i in range(5)]
        hierarchy=([1,2,3,4,5],[1,2,3,4,5],[0,1,2,0,1]);glyph=torch.ones(5,1,64,64)
        reference=sample_latents(echo,Once(),noise,conds,1.,1,'cpu',skel=glyph,hier_conds=hierarchy)
        for batch in (2,3):
            actual=sample_latents(echo.cuda(),Once(),noise,conds,1.,batch,'cuda',skel=glyph,hier_conds=hierarchy)
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        records['cuda_tail_5_samples_batches_2_3_vs_single']=True
    if a.out:Path(a.out).write_text(json.dumps(records,indent=2))
    print('PREFLIGHT',records,flush=True)


if __name__=='__main__':main()
