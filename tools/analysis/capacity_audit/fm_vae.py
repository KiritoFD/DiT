"""No optimizer: matched GT/noise FM buckets and deterministic VAE round-trip."""
import argparse
from collections import Counter, defaultdict
import csv
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from PIL import Image
from tools.analysis.capacity_audit.common import rows, sha, write, summarize, structure, vae_identity
from src.eval.in_mem_eval import _get_cache
from src.eval.model_io import load_model_from_ckpt
from src.eval.sampler import model_kwargs_for_slice
from src.utils.vae_scaling import resolve_vae_scaling
from src.loss.flow_matching import TIME_SCALE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', action='append', required=True, help='label=path')
    p.add_argument('--out', required=True);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--seeds',type=int,default=2);p.add_argument('--memory-fraction',type=float,default=.075)
    p.add_argument('--reuse',default='',help='Prior audit directory: identical conditions and sampled GT latents')
    p.add_argument('--train-csv',default='exp-std/csv/train_top10_aug_sym.csv')
    p.add_argument('--val-csv',default='exp-std/csv/eval200_fixed.csv')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    train=rows(a.train_csv); val=rows(a.val_csv)
    for key in ('image_path','src_image_path'):
        assert not ({r[key] for r in train if r.get(key)} & {r[key] for r in val if r.get(key)}), key
    pool=defaultdict(list)
    for r in train:
        if not r.get('aug'): pool[r['calligrapher'],r['script']].append(r)
    rng=np.random.default_rng(9017);chosen=[]
    for slot, n in Counter((r['calligrapher'],r['script']) for r in val).items():
        candidates=pool[slot]
        assert len(candidates)>=n, slot
        chosen.extend(candidates[i] for i in rng.choice(len(candidates),n,replace=False))
    combined=chosen+val; conditions=out/'conditions.csv'
    with conditions.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in combined))))
        w.writeheader();w.writerows(combined)
    checkpoints=dict(s.split('=',1) for s in a.checkpoint)
    ck=torch.load(next(iter(checkpoints.values())),map_location='cpu',weights_only=False)
    args=ck['args']; del ck
    sf,shift=resolve_vae_scaling(args)
    vae_files=vae_identity(args.vae_path)
    cache=_get_cache(str(conditions),len(combined),None,'',args)
    if a.reuse:
        reuse=Path(a.reuse)
        assert sha(conditions)==sha(reuse/'conditions.csv'), 'Different sample order'
        prior=json.loads((reuse/'vae.json').read_text(encoding='utf-8'))
        assert (prior['sf'],prior['shift'])==(sf,shift)
        assert prior['vae_files']==vae_files, 'Different VAE weights/config'
        z=torch.load(reuse/'gt_latents.pt',map_location='cpu',weights_only=True)
        assert z.shape==(len(combined),4,32,32)
        torch.save(z,out/'gt_latents.pt')
        write(out/'vae.json',dict(prior,reused_from=str(reuse)))
    else:
        z=encode_and_reconstruct(args,cache,chosen,val,combined,a,out,sf,shift)
    run_models(a,out,checkpoints,combined,chosen,val,train,conditions,cache,z,sf,shift)


def encode_and_reconstruct(args,cache,chosen,val,combined,a,out,sf,shift):
    from diffusers import AutoencoderKL
    vae=AutoencoderKL.from_pretrained(args.vae_path).cuda().eval().requires_grad_(False)
    latent=[]; recon_records=[];(out/'recon').mkdir()
    from src.eval.metrics import ssim
    from src.eval.metrics_ink import ink_iou
    # Fixed posterior samples give all models exactly the same training targets;
    # posterior means separately diagnose deterministic reconstruction.
    gen=torch.Generator(device='cuda').manual_seed(9017)
    for start in range(0,len(combined),a.batch):
        x=cache['gts'][start:start+a.batch].cuda()
        posterior=vae.encode(x).latent_dist
        latent.append(((posterior.sample(generator=gen)-shift)*sf).cpu())
        if start+a.batch>len(chosen):
            recon=vae.decode(posterior.mean).sample.clamp(-1,1).cpu()
            for j in range(len(x)):
                i=start+j-len(chosen)
                if i<0: continue
                pred=((recon[j]+1)/2).permute(1,2,0).numpy()
                gt=((x[j].cpu()+1)/2).permute(1,2,0).numpy()
                Image.fromarray((pred*255).astype('uint8')).save(out/'recon'/f'{i}.png')
                recon_records.append({'index':i,'character':val[i]['character'],
                    'ssim':ssim(pred,gt),'ink_iou':ink_iou(pred,gt),**structure(pred,gt)})
    z=torch.cat(latent);torch.save(z,out/'gt_latents.pt')
    write(out/'vae.json',{'sf':sf,'shift':shift,'posterior_reconstruction':'mean',
        'vae_files':vae_identity(args.vae_path),
        'posterior_fm':'fixed sample seed9017','records':recon_records,
        'limitation':'Geometry signals are not automatic labels of genuine flying-white or OCR errors.'})
    del vae,x,posterior;gc.collect();torch.cuda.empty_cache()
    return z


def run_models(a,out,checkpoints,combined,chosen,val,train,conditions,cache,z,sf,shift):
    seen={r['character'] for r in train}
    report={'models':{},'n_train':len(chosen),'n_val':len(val),'seeds':a.seeds,
        'train_csv_sha256':sha(a.train_csv),'val_csv_sha256':sha(a.val_csv),
        'times':[.1,.25,.4,.6,.8,.95],'sf':sf,'shift':shift,
        'latent_sha256':sha(out/'gt_latents.pt'),
        'latent_values_sha256':__import__('hashlib').sha256(z.numpy().tobytes()).hexdigest(),
        'conditions_sha256':sha(conditions),
        'limitations':['Teacher-forced local FM, not free-generation quality.',
         'Train subset matches validation writer/script counts; original non-augmented images only.',
         'Train and val have separate samples; models share identical targets and noise.',
         'No source-domain holdout claim; no STD augmentation applied; no optimizer.',
         'Repeated seeds/times are not independent samples. Raw records permit grouped inference.']}
    for label,path in checkpoints.items():
        model,ma=load_model_from_ckpt(path,device='cuda',use_ema=True)
        model.eval().requires_grad_(False)
        assert Path(ma.data_csv).resolve()==Path(a.train_csv).resolve(), 'Different declared training CSV'
        assert resolve_vae_scaling(ma)==(sf,shift)
        assert vae_identity(ma.vae_path)==json.loads((out/'vae.json').read_text(encoding='utf-8'))['vae_files']
        mc=_get_cache(str(conditions),len(combined),None,'',ma)
        assert torch.equal(mc['skels_latent'],cache['skels_latent']), 'Different STD contract'
        records=[]
        for seed in range(a.seeds):
            noise=torch.randn(z.shape,generator=torch.Generator().manual_seed(9018+seed))
            for t in report['times']:
                for start in range(0,len(combined),a.batch):
                    end=min(start+a.batch,len(combined));idx=list(range(start,end))
                    kw=model_kwargs_for_slice(model,yc=torch.tensor([mc['conds'][i][0] for i in idx],device='cuda'),
                        yh=torch.tensor([mc['conds'][i][1] for i in idx],device='cuda'),
                        hier_slice=tuple([axis[i] for i in idx] for axis in mc['hier_conds']),
                        skel_slice=mc['skels_latent'][idx],device='cuda')
                    gt=z[idx].cuda();eps=noise[idx].cuda()
                    pred=model((1-t)*gt+t*eps,torch.full((len(idx),),t*TIME_SCALE,device='cuda'),**kw)
                    if isinstance(pred,tuple):pred=pred[0]
                    mse=(pred[:,:4].float()-(eps-gt)).square().flatten(1).mean(1).cpu().tolist()
                    for i,v in zip(idx,mse):
                        r=combined[i]
                        records.append({'index':i,'sample_id':r['img_id'],'split':'train' if i<len(chosen) else 'val',
                            'writer':r['calligrapher'],'script':r['script'],'character':r['character'],
                            'character_seen':r['character'] in seen,'seed':seed,'t':t,'mse':v})
                print(label,seed,t,'done',flush=True)
        write(out/f'{label}_records.json',records)
        report['models'][label]={'path':path,'sha256':sha(path),'buckets':summarize(records)}
        write(out/'report.json',report)
        del model,mc,pred;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
