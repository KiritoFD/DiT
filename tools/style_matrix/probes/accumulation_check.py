"""One fixed 256-sample update at micro 256/128/64, including clip/Adam/EMA/LR."""
import argparse
import copy
import json
from pathlib import Path
import torch
from src.unet.blocks import ResBlock
from src.unet.io import diffusion
from src.train.ema import update_ema
from src.train.step import build_cosine_scheduler
from src.unet.style.diagnostics import inspect
from ..runtime.common import construct,optimizer,synthetic,step,seed,save


def relative(a,b):
    numerator=sum((a[k]-b[k]).double().square().sum().item() for k in a)
    denominator=sum(a[k].double().square().sum().item() for k in a)
    return (numerator/max(denominator,1e-30))**.5


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fp32',action='store_true')
    p.add_argument('--batch',type=int,default=256)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text());model,cfg,_=construct(spec,initialize_source=False)
    torch.set_num_threads(8)
    if a.fp32:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for module in model.backbone.modules():
        if isinstance(module,ResBlock):torch.nn.init.normal_(module.conv2.weight,std=.001)
    torch.nn.init.normal_(model.backbone.output[-1].weight,std=.001)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32):inspect(model,synthetic(cfg),calibrate=True)
    initial={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    inputs=synthetic(cfg,a.batch);z,_,g,w,s=inputs;reference=None;reports=[]
    choices=(512,448,256,224,128) if a.batch==512 else (256,192,128,64)
    for micro in choices:
        model.load_state_dict(initial,strict=True);model.train()
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        opt=optimizer(model,cfg);scheduler=build_cosine_scheduler(opt,warmup_steps=300,total_steps=36000,min_lr_ratio=.1)
        opt.zero_grad(set_to_none=True);seed(314159)
        loss=step(model,diffusion(cfg),z,g,w,s,micro,amp=not a.fp32)
        gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step();update_ema(ema,model,.999)
        updates={n:p.detach().cpu()-initial[n] for n,p in model.named_parameters() if p.requires_grad}
        ema_updates={n:p.detach().cpu()-initial[n] for n,p in ema.named_parameters() if n in updates}
        assert scheduler.last_epoch==1
        assert all(int(value['step'])==1 for value in opt.state.values())
        result=dict(microbatch=micro,accum=(a.batch+micro-1)//micro,fm=float(loss),optimizer_steps=1,
                    scheduler_steps=1,ema_updates=1)
        if reference is None:reference=(gradients,updates,ema_updates,float(loss))
        else:
            result.update(relative_gradient_error=relative(reference[0],gradients),
                          relative_update_error=relative(reference[1],updates),
                          relative_ema_error=relative(reference[2],ema_updates),
                          relative_loss_error=abs(float(loss)-reference[3])/abs(reference[3]))
            assert result['relative_gradient_error']<(.0001 if a.fp32 else .03), result
            assert result['relative_update_error']<(.005 if a.fp32 else .1), result
            assert result['relative_loss_error']<.002, result
        reports.append(result);print(json.dumps(result),flush=True)
        del opt,ema,gradients,updates,ema_updates
    save(a.out,dict(passed=True,precision='FP32, TF32 disabled' if a.fp32 else 'BF16 forward, FP32 master/Adam',batch=a.batch,
                    capacity_only=True,tolerances=dict(gradient_relative_L2=.0001 if a.fp32 else .03,
                        update_relative_L2=.005 if a.fp32 else .1,loss_relative=.002),results=reports))


if __name__=='__main__':main()
