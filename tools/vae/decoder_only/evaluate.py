"""Paired original/new decoder evaluation on identical GT and saved P2 latents."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

from .train import Images, frozen_hash


def describe(x, gt):
    gray=x.mean(0); truth=gt.mean(0)
    d=dict(l1=float(np.abs(x-gt).mean()),psnr=float(-10*np.log10(np.mean((x-gt)**2)+1e-12)),
           ssim=float(structural_similarity(gray,truth,data_range=1.)),soft_ink=float((1-gray).mean()))
    for thr in (.35,.45,.5,.55,.65):
        b=(1-gray)>thr; target=(1-truth)>thr
        labels,n=ndi.label(b); sizes=np.bincount(labels.ravel())[1:]
        d[f'frag_{thr}']=int((sizes>=4).sum())
        d[f'ink_iou_{thr}']=float((b&target).sum()/max(1,(b|target).sum()))
    gy,gx=np.gradient(gray); d['edge_strength']=float(np.hypot(gx,gy).mean())
    ink=1-gray; yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]]
    mask=ink>.5; widths=2*ndi.distance_transform_edt(mask)[skeletonize(mask)]
    d['width_p50']=float(np.median(widths)) if len(widths) else 0.
    d['cx']=float((ink*xx).sum()/max(1e-9,ink.sum())/gray.shape[1])
    d['cy']=float((ink*yy).sum()/max(1e-9,ink.sum())/gray.shape[0])
    return d


def main():
    p=argparse.ArgumentParser(); p.add_argument('--p2',required=True)
    p.add_argument('--vae',required=True); p.add_argument('--out',required=True)
    p.add_argument('--bank',default='assets/p2_audit_20261025/cfg/cfg1.0')
    a=p.parse_args()
    from src.unet.io import checkpoint_config
    from diffusers import AutoencoderKL
    import lpips
    cfg=checkpoint_config(torch.load(a.p2,map_location='cpu',weights_only=False))
    bank=Path(a.bank); identity=json.loads((bank/'identity.json').read_text())
    sha=hashlib.sha256()
    with open(a.p2,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): sha.update(block)
    assert identity['checkpoint_sha256']==sha.hexdigest(), 'P2 latent bank identity mismatch'
    rows=json.loads((bank/'samples.json').read_text())
    train=list(csv.DictReader(open(cfg.data_csv,encoding='utf-8-sig')))
    seen_chars={r['character'] for r in train}
    seen_triples={(r['calligrapher_id'],r['script_id'],r['character']) for r in train}
    latent=torch.from_numpy(np.load(bank/'latents.npy')).float()
    assert len(rows)==len(latent)
    orig=AutoencoderKL.from_pretrained(cfg.vae_path,local_files_only=True).cuda().eval().requires_grad_(False)
    new=AutoencoderKL.from_pretrained(a.vae,local_files_only=True).cuda().eval().requires_grad_(False)
    assert frozen_hash(orig)==frozen_hash(new), 'Decoder-only contract violated'
    lp=lpips.LPIPS(net='vgg',verbose=False).cuda().eval().requires_grad_(False)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    all_stats=[]; poster=Image.new('RGB',(5*256,16*280),'white'); draw=ImageDraw.Draw(poster)
    def png(x): return Image.fromarray((x.transpose(1,2,0)*255).round().clip(0,255).astype('uint8'))
    for i,row in enumerate(rows):
        im=Image.open(row['image_path']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
        x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None].cuda().float()/127.5-1
        with torch.no_grad():
            posterior=orig.encode(x).latent_dist
            torch.testing.assert_close(posterior.parameters,new.encode(x).latent_dist.parameters,rtol=0,atol=0)
            z=posterior.mode()
            zg=latent[i:i+1].cuda()/cfg.vae_scaling_factor+cfg.vae_shift_factor
            imgs=[x,orig.decode(z).sample,new.decode(z).sample,
                  orig.decode(zg).sample,new.decode(zg).sample]
            gt=((x+1)/2).clamp(0,1)[0].cpu().numpy()
            for label,t in zip(('GT','recon_original','recon_new','P2_original','P2_new'),imgs):
                arr=((t+1)/2).clamp(0,1)[0].cpu().numpy()
                metrics=describe(arr,gt)
                metrics['lpips']=float(lp(t,x).mean())
                all_stats.append(dict(index=i,kind=label,character=row['character'],
                    writer=row['calligrapher_id'],script=row['script_id'],
                    p2_unseen_character=row['character'] not in seen_chars,
                    p2_unseen_triple=(row['calligrapher_id'],row['script_id'],row['character']) not in seen_triples,
                    metrics=metrics))
                folder=out/label; folder.mkdir(exist_ok=True)
                png(arr).save(folder/f'{i:04d}.png')
                if i<16:
                    col=('GT','recon_original','recon_new','P2_original','P2_new').index(label)
                    poster.paste(png(arr),(col*256,i*280+24))
                    draw.text((col*256+4,i*280+4),f'{i} {label}',fill='black')
        if i%20==0: print('EVAL',i,len(rows),flush=True)
    summary={k:{m:float(np.mean([r['metrics'][m] for r in all_stats if r['kind']==k]))
                 for m in all_stats[0]['metrics']} for k in ('GT','recon_original','recon_new','P2_original','P2_new')}
    strata={}
    for flag in ('p2_unseen_character','p2_unseen_triple'):
        subset=[r for r in all_stats if r[flag]]
        strata[flag]=dict(n=sum(r['kind']=='GT' for r in subset),metrics={
            k:{m:float(np.mean([r['metrics'][m] for r in subset if r['kind']==k]))
               for m in all_stats[0]['metrics']} for k in summary} if subset else {})
    (out/'per_image.json').write_text(json.dumps(all_stats,ensure_ascii=False))
    (out/'summary.json').write_text(json.dumps(dict(n=len(rows),p2_sha256=sha.hexdigest(),
        bank=str(bank),frozen_hash_equal=True,posterior_bitwise_equal=True,metrics=summary,
        strata=strata,unseen_scope='relative to P2 training CSV, not all pretraining',
        character_correctness='requires visual review; no uncalibrated OCR score claimed'),indent=2))
    poster.save(out/'comparison.png'); print(json.dumps(summary),flush=True)
    # Same STD/noise/weights and exactly cached z for all writer-difference
    # probes. The baseline shared panel is generated once, then only decoded.
    del orig,new,lp
    import gc
    gc.collect()
    from tools.stroke_interaction.f4_gt_vs_gen import main as shared_probe
    shared=['--ckpt',a.p2,'--csv',cfg.data_csv,'--latent-cache',str(out.parent/'shared_latents'),
            '--chars','40','--per-kind','20','--seed','11']
    original_out=out.parent/'shared_original'
    if not (original_out/'gt_vs_generated.json').exists():
        shared_probe(shared+['--out',str(original_out)])
    shared_probe(shared+['--vae-path',a.vae,'--out',str(out/'shared_writer')])


if __name__=='__main__': main()
