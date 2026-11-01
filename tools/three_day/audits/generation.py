"""Deep114 on 48: exact native EMA generation bank, 48 characters x 3 seeds."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tools.three_day.contracts import read,save,sha,checkpoint_rows
from .teachers import decode


def main(root):
    from tools.u100_pretrain.eval_offline import load_legacy_ema_verified
    from src.unet.io import eval_args,diffusion
    from src.eval.in_mem_eval import _get_cache
    from src.eval.sampler import sample_latents
    from src.eval.decoder import load_eval_vae
    root=Path(root);out=root/'bank';out.mkdir(exist_ok=True)
    ckpt=Path('assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt')
    model,c,ck,verified=load_legacy_ema_verified(ckpt);model.eval().requires_grad_(False)
    mapping=checkpoint_rows(c);rows=read(c.eval_csv);cache=_get_cache(c.eval_csv,len(rows),None,'',eval_args(c))
    indices=[];seen=set()
    for i,r in enumerate(rows):
        if r['character'] not in seen:indices.append(i);seen.add(r['character'])
        if len(indices)==48:break
    if len(indices)!=48:raise ValueError('Need 48 distinct audit characters')
    selected=[rows[i] for i in indices];metadata=[];zs=[]
    torch.save(1-cache['skels_latent'][indices].float(),out/'std_gray.pt')
    for seed in (0,1,2):
        noise=torch.randn((48,4,32,32),generator=torch.Generator().manual_seed(seed))
        conds=[cache['conds'][i] for i in indices];hier=tuple([[axis[i] for i in indices] for axis in cache['hier_conds']])
        for r,cond in zip(selected,conds):
            if cond[0]!=mapping[int(r['calligrapher_id'])]:raise ValueError('Native cached Writer mapping mismatch')
        with torch.inference_mode():
            zs.append(sample_latents(model,diffusion(c),noise,conds,1.,16,'cuda',skel=cache['skels_latent'][indices],seed=seed,hier_conds=hier))
        metadata.extend([dict(r,seed=seed,source_eval_index=i) for i,r in zip(indices,selected)])
    z=torch.cat(zs);torch.save(z,out/'latents.pt');save(out/'metadata.json',metadata)
    save(out/'identity.json',dict(checkpoint=str(ckpt),sha256=sha(ckpt),semantic_writer_rows=mapping,
        legacy_ema=verified,scale=c.vae_scaling_factor,shift=c.vae_shift_factor,vae=c.vae_path,
        sigma=c.std_soft_eval_sigma,seeds=[0,1,2],n=144,characters=48,steps=c.eval_steps,cfg=1,
        std_sha256=__import__('hashlib').sha256(cache['skels_latent'][indices].numpy().tobytes()).hexdigest()))
    del model,ck;vae=load_eval_vae('cuda',c.vae_path).eval().requires_grad_(False)
    gt=cache['gts'][indices].float();recons=[]
    with torch.inference_mode():
        for start in range(0,48,8):
            x=gt[start:start+8].cuda();zg=(vae.encode(x).latent_dist.mean-c.vae_shift_factor)*c.vae_scaling_factor
            recons.append(decode(vae,zg,c.vae_scaling_factor,c.vae_shift_factor).cpu())
        images=torch.cat([decode(vae,z[i:i+8].cuda(),c.vae_scaling_factor,c.vae_shift_factor).cpu() for i in range(0,len(z),8)])
    torch.save(dict(gt=(gt+1)/2,reconstruction=torch.cat(recons),generated=images),out/'images.pt')
    for kind,xx in [('generated',images),('gt',(gt+1)/2),('reconstruction',torch.cat(recons))]:
        directory=out/kind;directory.mkdir(exist_ok=True)
        for i,x in enumerate(xx):Image.fromarray((x.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).save(directory/f'g{i}.png')
    print('NATIVE_BANK_DONE',len(z),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();torch.set_num_threads(8);main(a.root)
