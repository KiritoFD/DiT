"""Cache component-specific posterior distributions and text features, not Turbo latents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def setup(root):
    sys.path.insert(0,str(root))
    sys.path.insert(0,'/home/ds/Workspace/xy/qwen-calli-20261022/vendor')
    sys.path.insert(0,str(root/'vendor'))


def text_prompt(char,writer=None,script=None):
    if writer is not None:
        return (f'书法家：{writer}。书体：{script}书。请用{writer}的{script}书风格书写汉字「{char}」。'
                '纯白背景、黑色毛笔单字，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。')
    return f'纯白背景、黑色毛笔楷书单字「{char}」，字形完整，单字居中且几乎铺满画幅，无边框、无印章、无落款。'


def text_key(row,data):
    if data.get('prompt_schema')=='writer_script_v1':
        return (row['character'],row['calligrapher'],row['script'])
    return row['character']


def main(root,phase,model=None,limit=None):
    setup(root)
    import torch
    from PIL import Image
    import numpy as np
    torch.set_num_threads(8)
    component='vae' if phase=='latents' else 'text_encoder'
    if model is None:
        while not (root/f'{component}_ready').exists():
            time.sleep(15)
        model=root/'base'
    import fcntl
    gpu_lock=(root/'gpu.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    data=json.loads((root/'data.json').read_text(encoding='utf-8'))
    rows=sum(data['datasets'].values(),[])
    if limit:rows=rows[:limit]
    cfg=json.loads((root/'config.json').read_text())
    status=dict(phase=phase,pid=os.getpid(),model=str(model),completed=0)
    def save():
        (root/f'cache_{phase}_status.json').write_text(json.dumps(status,indent=2))
    save()
    if phase=='latents':
        from diffusers import AutoencoderKLQwenImage21
        vae=AutoencoderKLQwenImage21.from_pretrained(model,subfolder='vae',torch_dtype=torch.bfloat16,
                                                   local_files_only=True).eval().requires_grad_(False).cuda()
        means=[];stds=[]; names=[]
        for start in range(0,len(rows),4):
            batch=rows[start:start+4]
            images=[]
            for row in batch:
                path=root/row['file']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
                    raise ValueError(f'GT changed: {path}')
                im=Image.open(path).convert('RGBA').resize((cfg['resolution'],)*2,Image.Resampling.LANCZOS)
                images.append(torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()/127.5-1)
            pixels=torch.stack(images).cuda().to(torch.bfloat16).unsqueeze(2)
            with torch.no_grad():dist=vae.encode(pixels).latent_dist
            means.append(dist.mean.cpu());stds.append(dist.std.cpu());names.extend(r['file'] for r in batch)
            status['completed']=len(names);save()
        result=dict(mean=torch.cat(means),std=torch.cat(stds),names=names,
                    latents_mean=list(vae.config.latents_mean),latents_std=list(vae.config.latents_std))
    else:
        from diffusers import QwenImage21Pipeline
        pipe=QwenImage21Pipeline.from_pretrained(model,transformer=None,vae=None,scheduler=None,
                                                dtype=torch.bfloat16,local_files_only=True).to('cuda')
        pipe.text_encoder.requires_grad_(False).eval()
        result={}
        keys=sorted({text_key(r,data) for r in rows})
        for start in range(0,len(keys),8):
            batch=keys[start:start+8]
            prompts=[text_prompt(*key) if isinstance(key,tuple) else text_prompt(key) for key in batch]
            with torch.no_grad():emb,mask,pad=pipe.encode_prompt(prompts,device='cuda')
            if pad.any():raise ValueError('Text-only condition unexpectedly contains image tokens')
            for i,key in enumerate(batch):
                result[key]=(emb[i,mask[i].bool()] if mask is not None else emb[i]).cpu()
            status['completed']=len(result);save()
    suffix='_probe' if model!=root/'base' else ''
    torch.save(result,root/f'{phase}{suffix}.pt')
    identity=dict(model=str(model),component=component,resolution=cfg['resolution'],
                  data_sha256=hashlib.sha256((root/'data.json').read_bytes()).hexdigest(),
                  component_config_sha256=hashlib.sha256((model/component/'config.json').read_bytes()).hexdigest(),
                  limit=limit)
    (root/f'{phase}{suffix}_identity.json').write_text(json.dumps(identity,indent=2))
    status.update(phase='ready',output=f'{phase}{suffix}.pt');save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['latents','text'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args();main(a.root.resolve(),a.phase,a.model,a.limit)
