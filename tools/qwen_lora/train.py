"""Qwen 2.1 native FM LoRA, prompt-only conditioning (writer/script/char in text), single GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from cache import setup,text_key


def main(root,probe=False,resume=None,config_path=None):
    setup(root)
    import fcntl
    gpu_lock=(root/'gpu.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    import numpy as np
    import torch
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict,set_peft_model_state_dict
    from diffusers import QwenImage21Transformer2DModel,QwenImage21Pipeline,FlowMatchEulerDiscreteScheduler
    torch.set_num_threads(8);torch.manual_seed(20261023);random.seed(20261023)
    torch.backends.cuda.matmul.allow_tf32=True
    config=json.loads((config_path or root/'config.json').read_text())
    data=json.loads((root/'data.json').read_text(encoding='utf-8'))
    if data.get('prompt_schema')!='writer_script_v1':
        raise ValueError('Training requires writer+script prompts in the text cache')
    suffix='_probe' if probe else ''
    model_path=Path('/home/ds/Workspace/xy/qwen-calli-20261022/model') if probe else root/'base'
    for phase in ('latents','text'):
        status=json.loads((root/f'cache_{phase}_status.json').read_text())
        if status['model']!=str(model_path) or status['phase']!='ready' or status['output']!=f'{phase}{suffix}.pt':
            raise ValueError(f'Wrong/not-ready {phase} cache')
    latent=torch.load(root/f'latents{suffix}.pt',map_location='cpu',weights_only=True)
    text=torch.load(root/f'text{suffix}.pt',map_location='cpu',weights_only=True)
    if latent['mean'].shape[-2:]!=(config['resolution']//16,)*2:
        raise ValueError('Wrong latent cache resolution')
    if not probe:
        allrows=sum(data['datasets'].values(),[])
        if set(latent['names'])!={r['file'] for r in allrows} or set(text)!={text_key(r,data) for r in allrows}:
            raise ValueError('Partial/mismatched cache cannot be used for training')
        for phase in ('latents','text'):
            identity=json.loads((root/f'{phase}_identity.json').read_text())
            if identity['data_sha256']!=hashlib.sha256((root/'data.json').read_bytes()).hexdigest():
                raise ValueError(f'{phase} cache dataset identity changed')
    indices={name:i for i,name in enumerate(latent['names'])}
    rows=[r for r in data['datasets']['train'] if r['file'] in indices]
    if not rows:raise ValueError('No training rows in latent cache')
    writer_pools={wid:[r for r in rows if r['writer_index']==wid] for wid in sorted({r['writer_index'] for r in rows})}
    model=QwenImage21Transformer2DModel.from_pretrained(model_path,subfolder='transformer',
                        torch_dtype=torch.bfloat16,local_files_only=True).requires_grad_(False).cuda()
    model.add_adapter(LoraConfig(r=config['rank'],lora_alpha=config['rank'],
                                init_lora_weights='gaussian',target_modules=['to_q','to_k','to_v','to_out.0']))
    if config.get('gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    model.train()
    lora=[p for p in model.parameters() if p.requires_grad]
    for p in lora:p.data=p.data.float()
    optimizer=torch.optim.AdamW([{'params':lora,'lr':config['lr']}],weight_decay=.01)
    start_step=0;resume_lrs=None
    if resume is not None:
        if probe:raise ValueError('Cannot resume a two-step probe')
        from safetensors.torch import load_file
        checkpoint_state=torch.load(resume/'state.pt',map_location='cpu',weights_only=False)
        if checkpoint_state['data_sha256']!=hashlib.sha256((root/'data.json').read_bytes()).hexdigest():
            raise ValueError('Resume dataset identity mismatch')
        if checkpoint_state['base']!=str(model_path):raise ValueError('Resume base model mismatch')
        for key in ('rank','resolution','batch','accumulation'):
            if checkpoint_state['config'][key]!=config[key]:raise ValueError(f'Resume changed {key}')
        saved={k.removeprefix('transformer.'):v for k,v in load_file(str(resume/'pytorch_lora_weights.safetensors')).items()}
        set_peft_model_state_dict(model,saved,adapter_name='default')
        current=get_peft_model_state_dict(model)
        if set(current)!=set(saved) or any(not torch.equal(current[k].detach().cpu(),v) for k,v in saved.items()):
            raise RuntimeError('LoRA resume was not exact')
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        resume_lrs=[g['lr'] for g in optimizer.param_groups]
        start_step=int(checkpoint_state['step'])
        torch.set_rng_state(checkpoint_state['torch_rng'])
        torch.cuda.set_rng_state(checkpoint_state['cuda_rng'])
        random.setstate(checkpoint_state['python_rng'])
        print('RESUME_VERIFIED',str(resume),'step',start_step,'optimizer_groups',len(optimizer.param_groups),flush=True)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_path,subfolder='scheduler',local_files_only=True)
    sigmas=scheduler.sigmas.to('cuda');timesteps=scheduler.timesteps.to('cuda')
    mean=torch.tensor(latent['latents_mean'],device='cuda').view(1,64,1,1,1)
    std=torch.tensor(latent['latents_std'],device='cuda').view(1,64,1,1,1)
    target_steps=2 if probe else config['max_steps']
    if target_steps<=start_step:raise ValueError('Target step must exceed resume step')
    accumulation=1 if probe else config['accumulation']
    batch=config['batch']
    state=dict(pid=os.getpid(),phase='training',probe=probe,step=start_step,max_steps=target_steps,
               lora_parameters=sum(p.numel() for p in lora),conditioning='prompt_only',
               base=str(model_path),microbatch=batch,accumulation=accumulation)
    if resume is not None:state.update(resumed_from=str(resume),start_step=start_step,resume_verified=True)
    def save_status():
        (root/('probe_status.json' if probe else 'train_status.json')).write_text(json.dumps(state,indent=2))
    save_status()
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad} if probe else None
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()

    def checkpoint(step):
        dest=root/'checkpoints'/f'{step:07d}';dest.mkdir(parents=True,exist_ok=False)
        QwenImage21Pipeline.save_lora_weights(str(dest),transformer_lora_layers=get_peft_model_state_dict(model))
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                        python_rng=random.getstate(),config=config,
                        writers=data['writers'],scripts=data['scripts'],base=str(model_path),
                        data_sha256=hashlib.sha256((root/'data.json').read_bytes()).hexdigest()),dest/'state.pt')

    def evaluate(step):
        from PIL import Image
        from src.eval.metrics import ssim,get_lpips
        from src.eval.metrics_ink import ink_iou
        model.eval()
        pipe=QwenImage21Pipeline.from_pretrained(model_path,transformer=model,text_encoder=None,
                                                dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        lp,error=get_lpips('cuda')
        if lp is None:raise RuntimeError(error)
        out=root/'evaluations'/f'{step:07d}';out.mkdir(parents=True,exist_ok=False)
        selected=[]
        for wi in range(len(data['writers'])):
            for si in range(len(data['scripts'])):
                selected.extend([r for r in data['datasets']['validation']
                                 if r['writer_index']==wi and r['script_index']==si][:config.get('eval_per_slot',2)])
        metrics=[]
        for row in selected:
            emb=text[text_key(row,data)][None].cuda()
            mask=torch.ones(emb.shape[:2],device='cuda',dtype=torch.bool)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=pipe(prompt_embeds=emb,prompt_embeds_mask=mask,height=config['resolution'],
                            width=config['resolution'],output_resolution=config['resolution'],
                            num_inference_steps=config.get('eval_steps',40),true_cfg_scale=1.,use_kv_cache=True,
                            generator=torch.Generator('cpu').manual_seed(90)).images[0]
            filename=f'U{ord(row["character"]):04X}_w{row["writer_index"]}_s{row["script_index"]}_{len(metrics)}.png'
            white=Image.new('RGBA',result.size,'white');white.alpha_composite(result.convert('RGBA'))
            result=white.convert('RGB').resize((256,256),Image.Resampling.LANCZOS);result.save(out/filename)
            pred=np.asarray(result,dtype=np.float32)/255
            gt=np.asarray(Image.open(root/row['file']).convert('RGB'),dtype=np.float32)/255
            ten=lambda a:torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda()*2-1
            with torch.inference_mode():distance=float(lp(ten(pred),ten(gt)).item())
            actual_steps=len(pipe.scheduler.timesteps)
            if actual_steps!=config.get('eval_steps',40):raise ValueError(f'Unexpected sampling steps: {actual_steps}')
            metrics.append(dict(character=row['character'],writer=row['calligrapher'],script=row['script'],gt=row['file'],steps=actual_steps,
                                ssim=ssim(pred,gt),lpips=distance,ink_iou=ink_iou(pred,gt)))
        (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
        pipe.vae.to('cpu');del pipe,lp
        # The shared metric singleton remains on GPU, bounded AlexNet memory.
        model.train()

    for step in range(start_step+1,target_steps+1):
        optimizer.zero_grad(set_to_none=True);total=0.
        offset=step-start_step
        for i,(group,peak) in enumerate(zip(optimizer.param_groups,[config['lr']])):
            if resume_lrs is None and 'initial_horizon' in config:
                horizon=config['initial_horizon']
                if step<=horizon:
                    group['lr']=peak*min(1.,step/config['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*step/horizon)))
                else:
                    local=step-horizon;warm=config['extension_warmup'];top=config['extension_peak']
                    if local<=warm:
                        group['lr']=peak*.1+(top-peak*.1)*local/warm
                    else:
                        fraction=(local-warm)/(target_steps-horizon-warm)
                        group['lr']=config['min_lr']+(top-config['min_lr'])*.5*(1+math.cos(math.pi*fraction))
            elif resume_lrs is None:
                group['lr']=peak*min(1.,step/config['warmup'])*(.1+.9*.5*(1+math.cos(math.pi*step/target_steps)))
            elif offset<=config['warmup']:
                group['lr']=resume_lrs[i]+(peak-resume_lrs[i])*offset/config['warmup']
            else:
                fraction=(offset-config['warmup'])/(target_steps-start_step-config['warmup'])
                floor=config['min_lr']
                group['lr']=floor+(peak-floor)*.5*(1+math.cos(math.pi*fraction))
        for _ in range(accumulation):
            chosen=[random.choice(writer_pools[random.choice(list(writer_pools))]) for _ in range(batch)]
            selected=torch.tensor([indices[row['file']] for row in chosen])
            mu=latent['mean'][selected].cuda().float();sd=latent['std'][selected].cuda().float()
            clean=(mu+sd*torch.randn_like(mu)-mean)/std
            clean=clean.to(torch.bfloat16);noise=torch.randn_like(clean)
            index=torch.randint(len(timesteps),(batch,),device='cuda');sigma=sigmas[index].view(batch,1,1,1,1)
            z=(1-sigma)*clean+sigma*noise
            sequences=[text[text_key(row,data)] for row in chosen]
            emb=torch.nn.utils.rnn.pad_sequence(sequences,batch_first=True).cuda()
            lengths=torch.tensor([len(v) for v in sequences],device='cuda')
            text_mask=torch.arange(emb.shape[1],device='cuda')[None]<lengths[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                h,w=z.shape[-2:];packed=QwenImage21Pipeline._pack_latents(z,batch,64,h,w)
                img_mask=torch.cat([torch.zeros(batch,emb.shape[1],device='cuda',dtype=torch.bool),
                                    torch.ones(batch,h*w//4,device='cuda',dtype=torch.bool)],dim=1)
                prediction=model(hidden_states=packed,encoder_hidden_states=emb,encoder_hidden_states_mask=text_mask,
                                 timestep=timesteps[index]/1000,img_shapes=[[(1,h,w)]]*batch,img_mask=img_mask,
                                 return_dict=False)[0][:,-packed.shape[1]:]
                prediction=QwenImage21Pipeline._unpack_latents(prediction,config['resolution'],config['resolution'],16)
                loss=(prediction.float()-(noise-clean).float()).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite FM loss')
            (loss/accumulation).backward();total+=loss.item()/accumulation
        if step==start_step+1:
            norm=sum(float(p.grad.float().square().sum()) for p in lora if p.grad is not None)**.5
            if not math.isfinite(norm) or norm==0:raise RuntimeError(f'Invalid lora gradient: {norm}')
            state['lora_initial_grad_norm']=norm
        torch.nn.utils.clip_grad_norm_(lora,1.,error_if_nonfinite=True)
        optimizer.step()
        if step%10==0 or step==start_step+1 or step==target_steps:
            torch.cuda.synchronize()
            state.update(step=step,loss=total,seconds_per_step=(time.monotonic()-started)/(step-start_step),
                         learning_rates=[g['lr'] for g in optimizer.param_groups],
                         peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            save_status();print(json.dumps(state),flush=True)
        if not probe and (step%config['save_every']==0 or step==target_steps):
            checkpoint(step);evaluate(step)
    if probe:
        changed=any(not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if p.requires_grad)
        if not changed:raise RuntimeError('LoRA optimizer did not change parameters')
        state['lora_weights_changed']=changed
    state['phase']='completed';save_status()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--probe',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    try:main(a.root.resolve(),a.probe,a.resume.resolve() if a.resume else None,a.config)
    except Exception as error:
        path=a.root/('probe_status.json' if a.probe else 'train_status.json')
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(phase='failed',error=repr(error));path.write_text(json.dumps(state,indent=2))
        raise
