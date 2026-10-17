"""Fixed STD/script/noise, ten-writer counterfactuals using the canonical sampler."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.model_io import load_model_from_ckpt
from src.eval.in_mem_eval import _get_cache
from src.eval.sampler import sample_latents,build_diffusion
from src.eval.style.brush import features_from_gray
from diffusers import AutoencoderKL


def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=5)
    p.add_argument('--vae',default='data/pretrained/sd-vae-ft-ema')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    model,args=load_model_from_ckpt(a.ckpt,device=a.device,use_ema=True)
    model.eval();assert model.std_soft_cond and model.content_cond=='std'
    args.std_soft_root='exp-std/data/std_soft_glyphs_v1'
    args.std_soft_eval_sigma=.65
    rows=list(csv.DictReader(open('exp-std/csv/eval200_fixed.csv',encoding='utf-8-sig')))
    cache=_get_cache('exp-std/csv/eval200_fixed.csv',187,None,'',args)
    mapping=json.loads(Path(args.callig_remap_json).read_text())
    train=list(csv.DictReader(open('exp-std/csv/train.csv',encoding='utf-8-sig')))
    writers={int(mapping[str(r['calligrapher_id'])]):r['calligrapher'] for r in train}
    assert set(writers)==set(range(10))
    supported={(r['calligrapher'],r['script']) for r in train}
    ids=[0,23,62,82,8,31,75,88]
    entries=[];result=[]
    dev=torch.device(a.device)
    vae=AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    diff=build_diffusion(50,diffusion_type='flow',flow_kwargs={'sampler':'heun','sde_gamma':0.})
    for idx in ids:
        g=cache['skels_latent'][idx:idx+1].repeat(10,1,1,1)
        noise=cache['noise'][idx:idx+1].repeat(10,1,1,1)
        conds=[(w,cache['conds'][idx][1]) for w in range(10)]
        h=cache['hier_conds']
        hier=([w for w in range(10)],[h[1][idx]]*10,[h[2][idx]]*10)
        z=sample_latents(model,diff,noise,conds,1.,a.batch,dev,skel=g,seed=0,hier_conds=hier)
        torch.save(z,out/f'latent_{idx}.pt')
        for w in range(10):
            with torch.autocast(dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):
                x=vae.decode((z[w:w+1].to(dev)/.18215)).sample.float().cpu()
            rgb=((x[0].clamp(-1,1)+1)*127.5).permute(1,2,0).round().numpy().astype(np.uint8)
            path=out/f'g{len(entries)}.png';Image.fromarray(rgb).save(path)
            feat=features_from_gray(rgb)
            rr=dict(rows[idx]);rr['calligrapher']=writers[w]
            # IDs are only metadata for downstream judges; lookup remains explicit.
            rr['calligrapher_id']=next(k for k,v in mapping.items() if int(v)==w)
            entries.append(rr)
            result.append({'idx':len(entries)-1,'original_idx':idx,'writer_id':w,
                'writer':writers[w],'script':rr['script'],'character':rr['character'],
                'supported_slot':(writers[w],rr['script']) in supported,
                'ink':feat.scalars.get('ink_frac'),**feat.width_stats(),
                'noise_sha256':digest(noise[0].numpy().tobytes()),
                'g_sha256':digest(g[0].numpy().tobytes()),'png_sha256':digest(path.read_bytes())})
        print('DONE',idx,rows[idx]['character'],rows[idx]['script'],flush=True)
    with (out/'eval_map.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    manifest={'checkpoint':str(Path(a.ckpt).resolve()),'checkpoint_sha256':digest(Path(a.ckpt).read_bytes()),
        'weights':'ema','sampler':'heun','steps':50,'cfg':1.,'seed':0,'sde_gamma':0.,
        'sf':.18215,'shift':0.,'std_soft_sigma':.65,'writers':writers,'selected_eval_indices':ids,
        'n':len(entries),'per_image':result,
        'limitations':'each fixed character has one GT writer; other writer outputs have no paired GT target'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COUNTERFACTUAL_COMPLETE',out,flush=True)

if __name__=='__main__':main()
