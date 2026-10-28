"""Short, full-update capacity probes and causal invariants for generation two."""
import argparse
import copy
import json
import time
import torch
from src.unet.io import Config,build
from src.train.ema import update_ema


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--micro',type=int,default=256)
    p.add_argument('--steps',type=int,default=5);p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    cfg=Config(**json.load(open(a.spec))['config']);cfg.style_matrix_repr=-1;cfg.writer_velocity='cnn'
    model=build(cfg).cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(model).requires_grad_(False)
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg.lr,fused=True)
    x=torch.randn(a.micro,4,32,32,device='cuda');g=torch.rand(a.micro,1,64,64,device='cuda')
    w=torch.arange(a.micro,device='cuda')%10;s=w%3;t=x.new_full((a.micro,),750)
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        content,style=model.components(x[:2],t[:2],w[:2],g[:2],s[:2],True)
        null=model.components(x[:2],t[:2],torch.full_like(w[:2],10),g[:2],s[:2],True)
        assert torch.equal(content,null[0]);assert torch.count_nonzero(null[1])==0
    model.train();forward=torch.compile(model) if cfg.compile else model
    torch.cuda.reset_peak_memory_stats();times=[]
    for i in range(a.steps+2):
        torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
        for begin in range(0,cfg.batch,a.micro):
            n=min(a.micro,cfg.batch-begin)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=(forward(x[:n],t[:n],w[:n],g=g[:n],y_script=s[:n],condition_dropped=True).float()-x[:n]).square().mean()*n/cfg.batch
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        opt.step();update_ema(ema,model,.999);torch.cuda.synchronize()
        if i>=2:times.append(time.monotonic()-start)
    result=dict(parameters=sum(p.numel() for p in model.parameters()),
        branch_parameters=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('backbone.')),
        batch=cfg.batch,micro=a.micro,compile=cfg.compile,steps_s=len(times)/sum(times),
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,null_exact=True)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
