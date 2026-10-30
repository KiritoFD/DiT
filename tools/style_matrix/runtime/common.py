"""Shared matrix construction, optimizer groups, and portable identity helpers."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from src.unet.io import Config,build
from src.unet.style.transfer import initialize


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def construct(spec,device='cuda',initialize_source=True):
    config=Config(**spec['config']);seed(config.seed)
    model=build(config).to(device).to(memory_format=torch.channels_last)
    transfer={}
    if initialize_source and spec.get('initialization')!='scratch':
        if config.style_matrix_repr>=0:
            transfer=initialize(model,spec['source'],table=config.style_matrix_table)
            if spec.get('transfer_style'):
                ck=torch.load(spec['source'],map_location='cpu',weights_only=False)
                style={k.removeprefix('_orig_mod.'):v for k,v in ck['ema'].items()
                       if k.startswith(('style_representation.','style_injections.'))}
                report=model.load_state_dict(style,strict=False)
                if report.unexpected_keys:raise ValueError('Style transfer keys unknown: '+str(report.unexpected_keys))
                residue=[k for k in report.missing_keys
                         if k.startswith(('style_representation.','style_injections.'))]
                if residue:raise ValueError('Style transfer incomplete: '+str(residue))
                transfer.update(transfer_style=len(style),source_step_style=int(ck['train_steps']))
        else:
            ck=torch.load(spec['source'],map_location='cpu',weights_only=False)
            weights={k.removeprefix('_orig_mod.'):v for k,v in ck['ema'].items()}
            model.load_state_dict(weights,strict=True)
            transfer=dict(source=spec['source'],source_step=int(ck['train_steps']),strict_model_equal=True)
    if spec.get('initialization')=='scratch':
        if spec.get('source'):raise ValueError('Scratch study must not load a checkpoint')
        transfer=dict(initialization='scratch',seed=config.seed,source=None)
    return model,config,transfer


def optimizer(model,config,decoder=None,*,scratch=False):
    groups=[]
    if config.style_matrix_repr<0:
        groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=config.lr if scratch else 2e-5,name='unet')]
        if decoder is not None:groups.append(dict(params=list(decoder.parameters()),lr=1e-5,name='decoder',weight_decay=0.))
        return torch.optim.AdamW(groups,weight_decay=.01,fused=True)
    def rate(kind,legacy):
        if scratch:return config.lr
        explicit=getattr(config,'lr_'+kind,0.) or 0.
        return explicit or legacy
    split_table=(not scratch) and getattr(config,'lr_writer_table',0.)>0
    for name,legacy in (('backbone',2e-5),('style_representation',1e-4),('style_injections',1e-4)):
        parameters=[p for n,p in model.named_parameters()
                    if n.startswith(name+'.') and p.requires_grad
                    and not (split_table and name=='style_representation' and '.table.' in n)]
        groups.append(dict(params=parameters,lr=rate(name,legacy),name=name))
    if split_table:
        table=[p for n,p in model.named_parameters()
               if n.startswith('style_representation.') and '.table.' in n and p.requires_grad]
        groups.append(dict(params=table,lr=rate('writer_table',0.),name='writer_table'))
    return torch.optim.AdamW(groups,weight_decay=config.weight_decay,fused=True)


def synthetic(config,batch=8):
    generator=torch.Generator(device='cuda').manual_seed(12345)
    x=torch.randn(batch,4,32,32,generator=generator,device='cuda').to(memory_format=torch.channels_last)
    t=torch.full((batch,),750.,device='cuda')
    g=torch.rand(batch,1,64,64,generator=generator,device='cuda')
    w=torch.arange(batch,device='cuda')%10;s=w%3
    return x,t,g,w,s


@torch.no_grad()
def condition_report(model,config,calibrate=False):
    if config.style_matrix_repr>=0:
        from src.unet.style.diagnostics import inspect
        return inspect(model,synthetic(config),calibrate=calibrate)
    old=model.training;model.eval();x,t,g,w,s=synthetic(config)
    a=model(x,t,w,g=g,y_script=s);b=model(x,t,(w+1)%10,g=g,y_script=s)
    null=model(x,t,torch.full_like(w,model.num_writers),g=g,y_script=s)
    result=dict(swap_rms=float((a-b).float().square().mean().sqrt()),
                null_gap=float((a-null).float().square().mean().sqrt()),legacy_writer_intentionally_retained=True)
    model.train(old);return result


def step(model,process,z,g,w,s,micro,timing=None,amp=True,return_conditions=False):
    """One optimizer-step FM graph split into microbatches, identical random draws."""
    t=process.sample_t(len(z),z.device);noise=torch.randn_like(z)
    raw=getattr(model,'_orig_mod',model);base=getattr(raw,'backbone',raw)
    all_drop=torch.rand(len(z),device=z.device)<base.drop_all
    wd=all_drop|(torch.rand(len(z),device=z.device)<raw.drop_writer)
    sd=all_drop|(torch.rand(len(z),device=z.device)<base.drop_script)
    w=torch.where(wd,raw.num_writers,w);s=torch.where(sd,raw.num_scripts,s)
    losses=[]
    for begin in range(0,len(z),micro):
        part=slice(begin,begin+micro)
        if timing is not None:
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)];events[0].record()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            loss=process.training_losses(model,z[part],t[part],
                dict(g=g[part],y_callig=w[part],y_script=s[part],condition_dropped=True),
                noise=noise[part])['loss'].sum()/len(z)
        if timing is not None:events[1].record()
        loss.backward();losses.append(loss.detach())
        if timing is not None:
            events[2].record();timing.append(events)
    result=torch.stack(losses).sum()
    return (result,(w,s)) if return_conditions else result
