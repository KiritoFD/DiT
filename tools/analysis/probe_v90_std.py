"""Fixed-writer/noise clean/perturbed/wrong-character diagnostic. No optimizer."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache
from src.eval.sampler import sample_latents,build_diffusion,model_kwargs_for_slice
from src.eval.decoder import _lpips_per_sample
from src.train.std_shape import encoder_health
from src.utils.std_shape_aug import StdShapeAugment


@torch.no_grad()
def layer_probe(model,z,g,kw):
    records={};hooks=[]
    def callback(index):
        def record(module,args):
            x,c=args[:2]
            records[index]={'hidden_rms':float(x.float().square().mean().sqrt()),
                'condition_rms':float(c.float().square().mean().sqrt()),
                'adaln_rms':float(module.adaLN_modulation(c).float().square().mean().sqrt())}
        return record
    for index in sorted(set([0,3,6,9,12,15,len(model.blocks)-1])):
        if index<len(model.blocks):hooks.append(model.blocks[index].register_forward_pre_hook(callback(index)))
    try:model(z,torch.full((len(z),),500.,device=z.device),**kw)
    finally:
        for hook in hooks:hook.remove()
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--device',default='cuda')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    model,args=load_model_from_ckpt(a.checkpoint,device=a.device,use_ema=True)
    args.std_soft_root='exp-std/data/std_soft_glyphs_v1';args.std_soft_eval_sigma=.65
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    with open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    chosen=[];seen=set()
    for i,r in enumerate(rows):
        if r['character'] not in seen:chosen.append(i);seen.add(r['character'])
        if len(chosen)==a.count:break
    wrong=[next(j for j,r in enumerate(rows) if r['script']==rows[i]['script'] and r['character']!=rows[i]['character']) for i in chosen]
    bank=torch.load(a.cache,map_location='cpu',weights_only=True,mmap=True)
    lookup={tuple(k):j for j,k in enumerate(bank['keys'])}
    raw=bank['masks'][[lookup[int(rows[i]['script_id']),rows[i]['character']] for i in chosen]].to(a.device)
    clean=cache['skels_latent'][chosen].to(a.device)
    perturbed,stats=StdShapeAugment()(raw,clean,step=10000,
        generator=torch.Generator(device=a.device).manual_seed(90),force_tier=2)
    kw=model_kwargs_for_slice(model,yc=torch.tensor([cache['conds'][i][0] for i in chosen],device=a.device),
        yh=torch.tensor([cache['conds'][i][1] for i in chosen],device=a.device),
        hier_slice=tuple([[axis[i] for i in chosen] for axis in cache['hier_conds']]),skel_slice=clean,device=a.device)
    fixed=torch.randn((len(chosen),4,32,32),generator=torch.Generator().manual_seed(90)).to(a.device)
    result={'encoder_health':encoder_health(model,clean,fixed),
        'layer_probe':layer_probe(model,fixed,clean,kw),'rows':chosen,'wrong_rows':wrong,
        'notes':'Identical writer/script/noise per query. Wrong STD comes from a different character in the same script. No GT is an inference condition.'}
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained('data/pretrained/sd-vae-ft-ema').to(a.device).eval()
    diffusion=build_diffusion(50,diffusion_type='flow',flow_kwargs={'sampler':'heun','sde_gamma':0.})
    conditions=[cache['conds'][i] for i in chosen]
    hier=tuple([[axis[i] for i in chosen] for axis in cache['hier_conds']])
    outputs={}
    for name,g in [('clean',clean),('perturbed',perturbed),('wrong',cache['skels_latent'][wrong])]:
        z=sample_latents(model,diffusion,fixed.cpu(),conditions,1.,1,torch.device(a.device),skel=g.cpu(),seed=90,hier_conds=hier)
        images=[]
        for i in range(len(z)):
            decoded=vae.decode(z[i:i+1].to(a.device)/.18215).sample.float().clamp(-1,1)
            image=((decoded[0]+1)/2).permute(1,2,0).cpu().numpy();images.append(image)
            Image.fromarray((image*255).astype(np.uint8)).save(out/f'{name}_{i}.png')
        outputs[name]=np.stack(images)
    for name in ('perturbed','wrong'):
        values=_lpips_per_sample(outputs['clean'],outputs[name])
        if values is None:raise RuntimeError('Missing valid LPIPS-VGG')
        result[name+'_vs_clean_lpips']=values
    result['augmentation_fallback_fraction']=float(stats['fallback'].float().mean())
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
