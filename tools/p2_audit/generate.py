"""Native sampling only: fixed val100 CFG sweep and supported-script Writer swaps."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .common import setup,GuidedVelocity,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--csv',required=True)
    p.add_argument('--out',required=True);p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--window',type=float,nargs=2);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--swap',action='store_true');a=p.parse_args();torch.set_num_threads(8)
    model,c,ck,cache,rows,legal,process=setup(a.checkpoint,a.csv,a.out)
    if a.seed:
        cache['noise']=torch.randn(cache['noise'].shape,generator=torch.Generator().manual_seed(a.seed))
    noise=cache['noise'];g=cache['skels_latent'];conds=cache['conds'];hier=cache['hier_conds']
    if a.swap:
        selected=[];seen=set()
        for i,r in enumerate(rows):
            if r['character'] in seen:continue
            seen.add(r['character']);selected.append(i)
            if len(selected)==12:break
        indices=[];wanted=[]
        for i in selected:
            for w in legal[rows[i]['script']]:indices.append(i);wanted.append(w)
        noise=noise[indices];g=g[indices];rows=[dict(rows[i],calligrapher_id=str(w)) for i,w in zip(indices,wanted)]
        conds=[(w,cache['conds'][i][1]) for i,w in zip(indices,wanted)]
        hier=(wanted,wanted,[cache['hier_conds'][2][i] for i in indices])
    from src.eval.sampler import sample_latents
    guided=GuidedVelocity(model,a.scale,a.window).eval();start=time.monotonic()
    with torch.inference_mode():
        latent=sample_latents(guided,process,noise,conds,1.,16,'cuda',skel=g,seed=0,hier_conds=hier)
    elapsed=time.monotonic()-start
    out=Path(a.out);samples=out/'samples';samples.mkdir(exist_ok=True)
    np.save(out/'latents.npy',latent.numpy())
    from src.eval.decoder import load_eval_vae
    vae=load_eval_vae('cuda',c.vae_path)
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=vae.decode(latent[i:i+8].cuda()/c.vae_scaling_factor+c.vae_shift_factor).sample
            x=((x.float().clamp(-1,1)+1)*127.5).byte().cpu().permute(0,2,3,1).numpy()
            for j,im in enumerate(x):Image.fromarray(im).save(samples/f'g{i+j}.png')
    for i,r in enumerate(rows):
        if not a.swap:
            x=((cache['gts'][i].float().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
            Image.fromarray(x).save(samples/f'gt{i}.png')
    save(out/'samples.json',rows)
    save(out/'generation.json',dict(checkpoint=a.checkpoint,scale=a.scale,window=a.window,seed=a.seed,n=len(rows),
        seconds=elapsed,solver_nfe=process.nfe,network_calls=guided.network_calls,
        branch_sample_evals=guided.branch_sample_evals,equivalent_branch_nfe=guided.branch_sample_evals/len(rows),
        padding_counted=True,swap=a.swap,png_shape=[256,256,3],sampler='native FlowMatching Heun50'))
    print('GENERATED',a.out,len(rows),round(elapsed,1),flush=True)


if __name__=='__main__':main()
