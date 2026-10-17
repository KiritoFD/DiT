"""Matched-noise P0 sampling, plus an explicitly limited STD feature-ratio probe."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch


CASES={'baseline':(50,(1.,1.,1.),(1.,1.,1.)),
       'heun100':(100,(1.,1.,1.),(1.,1.,1.)),
       'skip08':(50,(.8,1.,1.),(1.,1.,1.)),
       'std08':(50,(1.,1.,1.),(.8,1.,1.))}


@torch.no_grad()
def std_ratios(model,args,device):
    from src.eval.in_mem_eval import _get_cache
    cache=_get_cache(args.eval_csv,187,args.img_root,'',args)
    projected={};records=[];handles=[]
    for i in range(3):
        def save(module,inputs,output,i=i):projected[i]=output.detach()
        def measure(module,inputs,i=i):
            term=model._scaled(projected[i],model.std_gates,i,model.inference_std_scales).float()
            h=inputs[0].float()-term
            records.append(dict(scale=(32,16,8)[i],std_rms=float(term.square().mean().sqrt()),
                                feature_rms=float(h.square().mean().sqrt()),
                                ratio=float(term.square().mean().sqrt()/h.square().mean().sqrt().clamp_min(1e-8))))
        handles.append(model.std_proj[i].register_forward_hook(save))
        handles.append(model.encoder[i].register_forward_pre_hook(measure))
    try:
        pairs=cache['conds'][:4]
        w=torch.tensor([p[0] for p in pairs],device=device)
        s=torch.tensor(cache['hier_conds'][2][:4],device=device)
        model(cache['noise'][:4].to(device),torch.full((4,),1000.,device=device),w,
              g=cache['skels_latent'][:4].to(device),y_script=s)
    finally:
        for handle in handles:handle.remove()
    return dict(t=1.,model_time=1000.,n=4,interpretation='Pure-noise endpoint probe, not a trajectory-wide dominance estimate',scales=records)


def sample_cases(checkpoint,out,states,cases,device='cuda'):
    from src.unet.io import load,eval_args,diffusion
    from src.eval.in_mem_eval import run_in_mem_eval
    from src.eval.protocol import sha256_file
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(checkpoint)
    manifest=json.loads(Path(str(checkpoint)+'.done').read_text())
    digest=sha256_file(str(checkpoint))
    assert manifest['sha256']==digest and manifest['size']==checkpoint.stat().st_size
    header=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if 'ema' in states and 'architecture' not in header:
        raise ValueError('20k legacy EMA is invalid; choose --states raw or a fixed-EMA checkpoint')
    del header
    result={}
    for state in states:
        model,config,ck=load(str(checkpoint),device,use_ema=state=='ema')
        step=int(ck['train_steps']);del ck
        for name in cases:
            steps,skip,std=CASES[name]
            cfg=replace(config,eval_steps=steps)
            args=eval_args(cfg);args.eval_use_ema=state=='ema'
            args.in_mem_eval_poster=False
            args.eval_protocol_extra=dict(case=name,skip=skip,std=std,ood=name in ('skip08','std08'))
            model.set_inference_scales(skip=skip,std=std)
            directory=out/f'{state}_{name}'
            directory.mkdir(parents=True,exist_ok=True)
            (directory/'provenance.json').write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()),
                file_sha256=digest,state=state,step=step,intervention=args.eval_protocol_extra,
                std_ratios=std_ratios(model,args,device)),indent=2))
            with torch.no_grad():
                values=run_in_mem_eval(model,args,step,device,results_dir=str(directory),diffusion_override=diffusion(cfg))
            generated=directory/f'eval_samples_ctrl/step{step:07d}/eval200fix'
            assert (generated/'DONE.json').exists()
            result[f'{state}_{name}']=dict(directory=str(generated),metrics=values,ood=args.eval_protocol_extra['ood'])
        del model
        gc.collect()
    (out/'cases.json').write_text(json.dumps(result,indent=2))
    return result
